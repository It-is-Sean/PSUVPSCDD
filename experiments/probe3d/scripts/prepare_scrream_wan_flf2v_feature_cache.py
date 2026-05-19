#!/usr/bin/env python3
"""Precompute SCRREAM pair features from WAN-FLF2V first/last-frame hidden states."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def ensure_torch_rmsnorm_compat() -> None:
    if hasattr(torch.nn, "RMSNorm"):
        return

    class RMSNorm(torch.nn.Module):
        def __init__(self, normalized_shape, eps: float | None = None, elementwise_affine: bool = True, device=None, dtype=None) -> None:
            super().__init__()
            if isinstance(normalized_shape, int):
                normalized_shape = (normalized_shape,)
            self.normalized_shape = tuple(normalized_shape)
            self.eps = torch.finfo(dtype or torch.float32).eps if eps is None else eps
            if elementwise_affine:
                self.weight = torch.nn.Parameter(torch.ones(self.normalized_shape, device=device, dtype=dtype))
            else:
                self.register_parameter("weight", None)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            dims = tuple(range(-len(self.normalized_shape), 0))
            variance = x.float().pow(2).mean(dim=dims, keepdim=True)
            output = x * torch.rsqrt(variance.to(dtype=x.dtype) + self.eps)
            if self.weight is not None:
                output = output * self.weight
            return output

    torch.nn.RMSNorm = RMSNorm  # type: ignore[attr-defined]


def ensure_torch_sdpa_compat() -> None:
    original = torch.nn.functional.scaled_dot_product_attention
    if getattr(original, "_wan_flf2v_compat", False):
        return

    def scaled_dot_product_attention_compat(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, **kwargs):
        enable_gqa = kwargs.pop("enable_gqa", False)
        scale = kwargs.pop("scale", None)
        if enable_gqa:
            raise RuntimeError("torch 2.3 scaled_dot_product_attention does not support enable_gqa=True")
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected SDPA kwargs for torch 2.3 compatibility wrapper: {unexpected}")
        return original(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale)

    scaled_dot_product_attention_compat._wan_flf2v_compat = True  # type: ignore[attr-defined]
    torch.nn.functional.scaled_dot_product_attention = scaled_dot_product_attention_compat


ensure_torch_rmsnorm_compat()
ensure_torch_sdpa_compat()


DEFAULT_ADAPTER_DATA = (
    "experiments/probe3d/adapter_data/"
    "scrream_mesh_complete_n2_adapter_seed17_tp20000_ms500000_trainplus_test.pt"
)
DEFAULT_CACHE_ROOT = "experiments/probe3d/feature_cache/scrream_wan_flf2v14b_pair_endpoint81_480"
DEFAULT_MODEL_ID = "checkpoints/wan2.1/Wan2.1-FLF2V-14B-720P-diffusers"
DEFAULT_TIMESTEPS = "249"
DEFAULT_LAYERS = "14"
NUM_FRAMES = 81
TEMPORAL_INDICES = (0, 20)
FEATURE_KIND_HIDDEN = "hidden"


@dataclass(frozen=True)
class WindowSpec:
    sample_id: str
    scene_id: str
    sequence_id: str
    frame_ids: tuple[int, int]
    frame_paths: tuple[str, str]
    sequence_dir: str
    window_mode: str
    synthetic_window: bool
    condition_frames: str
    slot_source_frame_ids: tuple[int, ...]
    window_paths: tuple[str, ...]
    temporal_indices: tuple[int, int]


def parse_csv_ints(value: str) -> list[int]:
    items = [int(x.strip()) for x in str(value).split(",") if x.strip()]
    if not items:
        raise ValueError(f"Expected non-empty CSV integer list, got {value!r}")
    return items


def safe_sample_id(sample_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", sample_id).strip("_")


def cache_path(cache_root: Path, timestep: int, layer: int, sample_id: str) -> Path:
    return cache_root / f"t{int(timestep):03d}" / f"layer{int(layer):02d}" / f"{safe_sample_id(sample_id)}.pt"


def load_rgb_paths(sequence_dir: Path) -> dict[int, Path]:
    rgb_dir = sequence_dir / "rgb"
    if not rgb_dir.is_dir():
        raise FileNotFoundError(f"Missing SCRREAM rgb directory: {rgb_dir}")
    mapping: dict[int, Path] = {}
    for path in sorted(rgb_dir.iterdir()):
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        try:
            frame_id = int(path.stem)
        except ValueError:
            continue
        mapping[frame_id] = path
    if not mapping:
        raise FileNotFoundError(f"No numbered RGB frames found under {rgb_dir}")
    return mapping


def build_window_spec(meta: dict[str, Any]) -> WindowSpec:
    frame_ids_raw = meta.get("frame_ids")
    frame_paths_raw = meta.get("frame_paths")
    if not frame_ids_raw or len(frame_ids_raw) != 2:
        raise ValueError(f"Sample {meta.get('sample_id', '<unknown>')} has invalid frame_ids={frame_ids_raw!r}")
    if not frame_paths_raw or len(frame_paths_raw) != 2:
        raise ValueError(f"Sample {meta.get('sample_id', '<unknown>')} has invalid frame_paths={frame_paths_raw!r}")

    frame_ids = (int(frame_ids_raw[0]), int(frame_ids_raw[1]))
    if frame_ids[1] < frame_ids[0]:
        raise ValueError(f"pair_endpoint81 expects ordered frame ids, got {frame_ids}")
    frame_paths = (str(frame_paths_raw[0]), str(frame_paths_raw[1]))
    sequence_dir = Path(frame_paths[0]).parent.parent
    rgb_map = load_rgb_paths(sequence_dir)
    sample_id = str(
        meta.get("sample_id")
        or f"{meta.get('scene_id')}/{meta.get('sequence_id')}_{frame_ids[0]:06d}_{frame_ids[1]:06d}"
    )
    scene_id = str(meta.get("scene_id", ""))
    sequence_id = str(meta.get("sequence_id", sequence_dir.name))

    slot_ids: list[int] = []
    for slot in range(NUM_FRAMES):
        alpha = slot / float(NUM_FRAMES - 1)
        frame_id = int(round(frame_ids[0] * (1.0 - alpha) + frame_ids[1] * alpha))
        slot_ids.append(frame_id)
    slot_ids[0] = frame_ids[0]
    slot_ids[-1] = frame_ids[1]
    missing = [idx for idx in sorted(set(slot_ids)) if idx not in rgb_map]
    if missing:
        raise FileNotFoundError(f"Missing RGB frames in {sequence_dir}: first missing ids {missing[:8]}")
    if str(rgb_map[frame_ids[0]]) != frame_paths[0] or str(rgb_map[frame_ids[1]]) != frame_paths[1]:
        pair_missing = [path for path in frame_paths if not Path(path).is_file()]
        if pair_missing:
            raise FileNotFoundError(f"Missing pair RGB frames for {sample_id}: {pair_missing}")

    return WindowSpec(
        sample_id=sample_id,
        scene_id=scene_id,
        sequence_id=sequence_id,
        frame_ids=frame_ids,
        frame_paths=frame_paths,
        sequence_dir=str(sequence_dir),
        window_mode="pair_endpoint81",
        synthetic_window=False,
        condition_frames="first_last",
        slot_source_frame_ids=tuple(slot_ids),
        window_paths=tuple(str(rgb_map[idx]) for idx in slot_ids),
        temporal_indices=TEMPORAL_INDICES,
    )


def load_frames(paths: tuple[str, ...]) -> list[Image.Image]:
    frames: list[Image.Image] = []
    for path in paths:
        with Image.open(path) as img:
            frames.append(img.convert("RGB").copy())
    return frames


def reshape_wan_tokens(raw: torch.Tensor, grid_shape: tuple[int, int, int]) -> torch.Tensor:
    t_tokens, h_tokens, w_tokens = grid_shape
    if raw.ndim != 3 or raw.shape[0] != 1:
        raise ValueError(f"Expected WAN hidden state shape [1,N,C], got {tuple(raw.shape)}")
    expected = t_tokens * h_tokens * w_tokens
    if raw.shape[1] != expected:
        raise ValueError(f"Expected {expected} WAN tokens for grid {grid_shape}, got {raw.shape[1]}")
    return raw.squeeze(0).reshape(t_tokens, h_tokens, w_tokens, raw.shape[-1]).contiguous()


class WanFLF2VFeatureExtractor:
    def __init__(
        self,
        model_id: str,
        prompt: str,
        device: str,
        dtype: str,
        height: int,
        width: int,
        scheduler: str,
        cpu_offload: bool,
    ) -> None:
        self.model_id = model_id
        self.prompt = prompt
        self.device = torch.device(device)
        self.height = int(height)
        self.width = int(width)
        self.scheduler_name = str(scheduler)
        if dtype == "bf16":
            self.torch_dtype = torch.bfloat16
        elif dtype == "fp16":
            self.torch_dtype = torch.float16
        elif dtype == "fp32":
            self.torch_dtype = torch.float32
        else:
            raise ValueError(f"Unsupported dtype={dtype!r}")
        self.pipe = self._load_pipeline(cpu_offload=bool(cpu_offload))
        with torch.no_grad():
            self.prompt_embeds, _ = self.pipe.encode_prompt(
                prompt=self.prompt,
                negative_prompt=None,
                do_classifier_free_guidance=False,
                num_videos_per_prompt=1,
                max_sequence_length=512,
                device=str(self.device),
            )
            self.prompt_embeds = self.prompt_embeds.to(device=self.device, dtype=self.pipe.transformer.dtype)
        if getattr(self.pipe, "text_encoder", None) is not None:
            self.pipe.text_encoder.to("cpu")
            torch.cuda.empty_cache()

    def _load_pipeline(self, cpu_offload: bool):
        try:
            from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
            from diffusers.models import WanTransformer3DModel
            from diffusers.models.modeling_outputs import Transformer2DModelOutput
            from diffusers.pipelines.wan.pipeline_wan_i2v import retrieve_latents
            from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
            from diffusers.utils import USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers
            from transformers import CLIPVisionModel
        except ImportError as exc:
            raise ImportError(
                "WAN FLF2V feature extraction requires diffusers, transformers, tokenizers, and accelerate."
            ) from exc

        class TransformerWanWithFeatureOutput(WanTransformer3DModel):
            def forward(
                self,
                hidden_states,
                timestep,
                encoder_hidden_states,
                encoder_hidden_states_image=None,
                return_dict=True,
                attention_kwargs=None,
                output_hidden_states=False,
                output_layers=None,
            ):
                if attention_kwargs is not None:
                    attention_kwargs = attention_kwargs.copy()
                    lora_scale = attention_kwargs.pop("scale", 1.0)
                else:
                    lora_scale = 1.0
                if USE_PEFT_BACKEND:
                    scale_lora_layers(self, lora_scale)

                batch_size, _, num_frames, height, width = hidden_states.shape
                p_t, p_h, p_w = self.config.patch_size
                post_patch_num_frames = num_frames // p_t
                post_patch_height = height // p_h
                post_patch_width = width // p_w
                rotary_emb = self.rope(hidden_states)
                hidden_states = self.patch_embedding(hidden_states)
                hidden_states = hidden_states.flatten(2).transpose(1, 2)
                if timestep.ndim == 2:
                    ts_seq_len = timestep.shape[1]
                    timestep = timestep.flatten()
                else:
                    ts_seq_len = None
                (
                    temb,
                    timestep_proj,
                    encoder_hidden_states,
                    encoder_hidden_states_image,
                ) = self.condition_embedder(
                    timestep,
                    encoder_hidden_states,
                    encoder_hidden_states_image,
                    timestep_seq_len=ts_seq_len,
                )
                if ts_seq_len is not None:
                    timestep_proj = timestep_proj.unflatten(2, (6, -1))
                else:
                    timestep_proj = timestep_proj.unflatten(1, (6, -1))
                if encoder_hidden_states_image is not None:
                    encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

                all_hidden_states = {}
                output_layers_set = set(output_layers or [])
                for idx, block in enumerate(self.blocks):
                    hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)
                    if output_hidden_states or idx in output_layers_set:
                        all_hidden_states[idx] = hidden_states.detach().clone()

                if temb.ndim == 3:
                    shift, scale = (self.scale_shift_table.unsqueeze(0).to(temb.device) + temb.unsqueeze(2)).chunk(2, dim=2)
                    shift = shift.squeeze(2)
                    scale = scale.squeeze(2)
                else:
                    shift, scale = (self.scale_shift_table.to(temb.device) + temb.unsqueeze(1)).chunk(2, dim=1)
                shift = shift.to(hidden_states.device)
                scale = scale.to(hidden_states.device)
                hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
                hidden_states = self.proj_out(hidden_states)
                hidden_states = hidden_states.reshape(
                    batch_size,
                    post_patch_num_frames,
                    post_patch_height,
                    post_patch_width,
                    p_t,
                    p_h,
                    p_w,
                    -1,
                )
                hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
                output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)
                if USE_PEFT_BACKEND:
                    unscale_lora_layers(self, lora_scale)
                if not return_dict:
                    return (output, all_hidden_states) if (output_hidden_states or output_layers) else (output,)
                if output_hidden_states or output_layers:
                    return Transformer2DModelOutput(sample=output), all_hidden_states
                return Transformer2DModelOutput(sample=output)

        class OneStepWanFLF2VPipeline(WanImageToVideoPipeline):
            @torch.no_grad()
            def __call__(self, video, image, last_image, t, output_layers, prompt_embeds, generator=None):
                device = self._execution_device
                transformer_dtype = self.transformer.dtype
                video_tensor = self.video_processor.preprocess_video(
                    video, height=self._probe_height, width=self._probe_width
                ).to(device, dtype=torch.float32)
                image_tensor = self.video_processor.preprocess(
                    image, height=self._probe_height, width=self._probe_width
                ).to(device, dtype=torch.float32)
                last_image_tensor = self.video_processor.preprocess(
                    last_image, height=self._probe_height, width=self._probe_width
                ).to(device, dtype=torch.float32)

                self.scheduler.set_timesteps(1000, device=device)
                timestep_index = int(t)
                if timestep_index < 0 or timestep_index >= len(self.scheduler.timesteps):
                    raise ValueError(f"WAN FLF2V timestep index must be in [0,{len(self.scheduler.timesteps) - 1}], got {timestep_index}")
                t_input = self.scheduler.timesteps[torch.tensor([timestep_index], dtype=torch.long, device=device)].to(device)

                latents = retrieve_latents(self.vae.encode(video_tensor), sample_mode="argmax")
                latents_mean = torch.tensor(self.vae.config.latents_mean).view(
                    1, self.vae.config.z_dim, 1, 1, 1
                ).to(latents.device, latents.dtype)
                latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
                    1, self.vae.config.z_dim, 1, 1, 1
                ).to(latents.device, latents.dtype)
                latents = (latents - latents_mean) * latents_std
                if generator is None:
                    noise = torch.randn(latents.shape, device=device, dtype=latents.dtype)
                else:
                    noise = torch.randn(latents.shape, device=device, dtype=latents.dtype, generator=generator)
                sigma = self.scheduler.sigmas[timestep_index].to(device=device, dtype=latents.dtype)
                noisy_latents = (1.0 - sigma) * latents + sigma * noise

                if self.transformer.config.image_dim is not None:
                    image_embeds = self.encode_image([image, last_image], device)
                    image_embeds = image_embeds.to(device=device, dtype=transformer_dtype)
                else:
                    image_embeds = None

                latents_out = self.prepare_latents(
                    image_tensor,
                    batch_size=1,
                    num_channels_latents=self.vae.config.z_dim,
                    height=self._probe_height,
                    width=self._probe_width,
                    num_frames=NUM_FRAMES,
                    dtype=torch.float32,
                    device=device,
                    generator=generator,
                    latents=noisy_latents,
                    last_image=last_image_tensor,
                )
                if self.config.expand_timesteps:
                    model_latents, condition, first_frame_mask = latents_out
                    latent_model_input = (1 - first_frame_mask) * condition + first_frame_mask * model_latents
                    timestep = (first_frame_mask[0][0][:, ::2, ::2] * t_input).flatten().unsqueeze(0)
                else:
                    model_latents, condition = latents_out
                    latent_model_input = torch.cat([model_latents, condition], dim=1)
                    timestep = t_input.expand(model_latents.shape[0])

                with self.transformer.cache_context("cond"):
                    transformer_result = self.transformer(
                        hidden_states=latent_model_input.to(dtype=transformer_dtype),
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds.to(device=device, dtype=transformer_dtype),
                        encoder_hidden_states_image=image_embeds,
                        return_dict=True,
                        output_layers=output_layers,
                    )
                if isinstance(transformer_result, tuple):
                    output, hidden_states = transformer_result
                else:
                    output, hidden_states = transformer_result, {}
                return output, hidden_states, {
                    "timestep_index": timestep_index,
                    "scheduler_timestep": float(t_input.detach().float().cpu().item()),
                    "sigma": float(sigma.detach().float().cpu().item()),
                    "scheduler_class": self.scheduler.__class__.__name__,
                    "scheduler_prediction_type": str(getattr(self.scheduler.config, "prediction_type", "")),
                    "source_latent_shape": list(latents.shape),
                    "transformer_latent_shape": list(latent_model_input.shape),
                    "model_output_shape": list(output.sample.shape),
                }

        transformer = TransformerWanWithFeatureOutput.from_pretrained(
            self.model_id,
            subfolder="transformer",
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
        )
        vae = AutoencoderKLWan.from_pretrained(
            self.model_id,
            subfolder="vae",
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        image_encoder = CLIPVisionModel.from_pretrained(
            self.model_id,
            subfolder="image_encoder",
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        pipe = OneStepWanFLF2VPipeline.from_pretrained(
            self.model_id,
            transformer=transformer,
            vae=vae,
            image_encoder=image_encoder,
            torch_dtype=self.torch_dtype,
        )
        if self.scheduler_name == "unipc":
            pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=3.0)
        elif self.scheduler_name != "checkpoint":
            raise ValueError(f"Unsupported scheduler={self.scheduler_name!r}")
        pipe._probe_height = self.height
        pipe._probe_width = self.width
        pipe.set_progress_bar_config(disable=True)
        if cpu_offload:
            pipe.enable_model_cpu_offload(device=str(self.device))
        else:
            pipe = pipe.to(self.device)
        pipe.transformer.eval()
        pipe.vae.eval()
        pipe.image_encoder.eval()
        pipe.transformer.requires_grad_(False)
        pipe.vae.requires_grad_(False)
        pipe.image_encoder.requires_grad_(False)
        pipe.text_encoder.eval()
        pipe.text_encoder.requires_grad_(False)
        return pipe

    @torch.no_grad()
    def extract(
        self,
        frames: list[Image.Image],
        timestep: int,
        layers: list[int],
        seed: int,
    ) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
        generator = torch.Generator(device=self.device)
        generator.manual_seed(int(seed))
        _, hidden_states, noise_info = self.pipe(
            video=frames,
            image=frames[0],
            last_image=frames[-1],
            t=int(timestep),
            output_layers=[int(x) for x in layers],
            prompt_embeds=self.prompt_embeds,
            generator=generator,
        )
        missing = set(layers) - set(hidden_states.keys())
        if missing:
            raise RuntimeError(f"WAN FLF2V did not return requested layers: {sorted(missing)}")
        features: dict[int, torch.Tensor] = {}
        for layer in layers:
            raw = hidden_states[int(layer)]
            feature_count = int(raw.shape[1])
            temporal = 21
            spatial = feature_count // temporal
            grid_height = max(1, self.height // 16)
            grid_width = max(1, self.width // 16)
            if temporal * grid_height * grid_width != feature_count:
                raise ValueError(
                    f"Unexpected FLF2V hidden token count {feature_count} for {self.height}x{self.width}; "
                    f"expected {temporal * grid_height * grid_width}"
                )
            features[int(layer)] = reshape_wan_tokens(raw, (temporal, grid_height, grid_width)).detach().cpu()
        return features, noise_info


def write_cache(path: Path, features: torch.Tensor, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save({"features": features.detach().cpu().to(torch.float16), "metadata": metadata}, tmp_path)
    tmp_path.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter_data", default=DEFAULT_ADAPTER_DATA)
    parser.add_argument("--output_dir", default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--model_id", default=DEFAULT_MODEL_ID, help="Local Diffusers checkpoint dir or HF model id.")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--timesteps", default=DEFAULT_TIMESTEPS)
    parser.add_argument("--layers", default=DEFAULT_LAYERS)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--fallback_height", type=int, default=720)
    parser.add_argument("--fallback_width", type=int, default=1280)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--split", default=None)
    parser.add_argument(
        "--sample_start",
        type=int,
        default=0,
        help="Start offset, inclusive, within the selected split before max_samples is applied.",
    )
    parser.add_argument(
        "--sample_end",
        type=int,
        default=0,
        help="End offset, exclusive, within the selected split. Use 0 to run through the end.",
    )
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16", choices=("bf16", "fp16", "fp32"))
    parser.add_argument(
        "--scheduler",
        default="checkpoint",
        choices=("checkpoint", "unipc"),
        help="FLF2V defaults to the checkpoint scheduler; unipc is kept only as an explicit diagnostic override.",
    )
    parser.add_argument("--cpu_offload", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--window_only", action="store_true")
    parser.add_argument("--log_every", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    timesteps = parse_csv_ints(args.timesteps)
    layers = parse_csv_ints(args.layers)
    if any(timestep < 0 or timestep > 999 for timestep in timesteps):
        raise ValueError(f"Timesteps must index the 1000-step scheduler in [0,999], got {timesteps}")
    if any(layer < 0 for layer in layers):
        raise ValueError(f"WAN FLF2V layer ids must be non-negative, got {layers}")
    if (args.height, args.width) not in {(480, 832), (720, 1280)}:
        logging.warning("Using nonstandard FLF2V probe resolution %sx%s", args.width, args.height)

    payload = torch.load(args.adapter_data, map_location="cpu")
    metadata = payload.get("metadata")
    splits = payload.get("splits", ["train" for _ in payload["scene_ids"]])
    if metadata is None:
        raise ValueError(f"{args.adapter_data} does not contain metadata")
    selected_indices = [idx for idx in range(len(metadata)) if args.split is None or splits[idx] == args.split]
    total_selected_count = len(selected_indices)
    if args.sample_start < 0:
        raise ValueError(f"sample_start must be non-negative, got {args.sample_start}")
    sample_end = int(args.sample_end) if args.sample_end and args.sample_end > 0 else total_selected_count
    if sample_end < args.sample_start:
        raise ValueError(f"sample_end must be >= sample_start, got {args.sample_start}:{sample_end}")
    selected_entries = list(enumerate(selected_indices))[int(args.sample_start) : sample_end]
    if args.max_samples and args.max_samples > 0:
        selected_entries = selected_entries[: args.max_samples]
    specs = [(selected_pos, build_window_spec(metadata[idx])) for selected_pos, idx in selected_entries]
    logging.info(
        "Validated %d WAN FLF2V pair_endpoint81 windows from %s (selected range %d:%d of %d)",
        len(specs),
        args.adapter_data,
        int(args.sample_start),
        sample_end,
        total_selected_count,
    )

    cache_root = Path(args.output_dir)
    grid_shape = [21, int(args.height) // 16, int(args.width) // 16]
    feature_shape = [2 * grid_shape[1] * grid_shape[2], None]
    manifest = {
        "adapter_data": str(args.adapter_data),
        "output_dir": str(cache_root),
        "model_id": str(args.model_id),
        "model_local_dir": str(args.model_id),
        "prompt": args.prompt,
        "timesteps": timesteps,
        "layers": layers,
        "feature_kind": FEATURE_KIND_HIDDEN,
        "feature_source": "wan_flf2v_hidden",
        "window_mode": "pair_endpoint81",
        "condition_frames": "first_last",
        "num_frames": NUM_FRAMES,
        "temporal_indices": list(TEMPORAL_INDICES),
        "height": int(args.height),
        "width": int(args.width),
        "fallback_height": int(args.fallback_height),
        "fallback_width": int(args.fallback_width),
        "grid_shape": grid_shape,
        "expected_feature_shape": feature_shape,
        "seed": int(args.seed),
        "split": args.split,
        "sample_start": int(args.sample_start),
        "sample_end": int(sample_end),
        "total_selected_count": int(total_selected_count),
        "sample_count": len(specs),
        "scheduler": args.scheduler,
        "source": "wan_flf2v_pair_endpoint81",
    }
    cache_root.mkdir(parents=True, exist_ok=True)
    manifest_name = "manifest.json"
    if int(args.sample_start) != 0 or sample_end != total_selected_count:
        manifest_name = f"manifest_samples_{int(args.sample_start):06d}_{sample_end:06d}.json"
    (cache_root / manifest_name).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if manifest_name != "manifest.json" and not (cache_root / "manifest.json").exists():
        full_manifest = dict(manifest)
        full_manifest["sample_start"] = 0
        full_manifest["sample_end"] = int(total_selected_count)
        full_manifest["sample_count"] = int(total_selected_count)
        full_manifest["sharded_cache"] = True
        (cache_root / "manifest.json").write_text(json.dumps(full_manifest, indent=2) + "\n", encoding="utf-8")
    if args.window_only:
        logging.info("window_only enabled; wrote manifest and skipped WAN loading.")
        return

    extractor: WanFLF2VFeatureExtractor | None = None
    written = 0
    skipped = 0
    for local_sample_idx, (selected_pos, spec) in enumerate(specs):
        frames: list[Image.Image] | None = None
        for timestep in timesteps:
            missing_layers = [
                layer
                for layer in layers
                if args.force or not cache_path(cache_root, timestep, layer, spec.sample_id).exists()
            ]
            if not missing_layers:
                skipped += len(layers)
                continue
            if extractor is None:
                logging.info("Loading WAN FLF2V model from %s", args.model_id)
                extractor = WanFLF2VFeatureExtractor(
                    model_id=str(args.model_id),
                    prompt=args.prompt,
                    device=args.device,
                    dtype=args.dtype,
                    height=args.height,
                    width=args.width,
                    scheduler=args.scheduler,
                    cpu_offload=bool(args.cpu_offload),
                )
            if frames is None:
                frames = load_frames(spec.window_paths)
            feature_seed = int(args.seed + selected_pos * 1000 + timestep)
            wan_features, noise_info = extractor.extract(frames, timestep=timestep, layers=missing_layers, seed=feature_seed)
            for layer, full_feature in wan_features.items():
                pair_feature = full_feature[list(spec.temporal_indices)].reshape(-1, full_feature.shape[-1]).contiguous()
                expected_tokens = 2 * (int(args.height) // 16) * (int(args.width) // 16)
                if pair_feature.shape[0] != expected_tokens:
                    raise ValueError(f"Expected pair token count {expected_tokens}, got {tuple(pair_feature.shape)}")
                out_path = cache_path(cache_root, timestep, layer, spec.sample_id)
                cache_meta = {
                    **asdict(spec),
                    "feature_kind": FEATURE_KIND_HIDDEN,
                    "feature_source": "wan_flf2v_hidden",
                    "model_id": str(args.model_id),
                    "model_local_dir": str(args.model_id),
                    "prompt": args.prompt,
                    "timestep": int(timestep),
                    "timestep_index": int(noise_info["timestep_index"]),
                    "scheduler_timestep": float(noise_info["scheduler_timestep"]),
                    "sigma": float(noise_info["sigma"]),
                    "scheduler_class": noise_info.get("scheduler_class"),
                    "scheduler_prediction_type": noise_info.get("scheduler_prediction_type"),
                    "layer": int(layer),
                    "seed": feature_seed,
                    "height": int(args.height),
                    "width": int(args.width),
                    "grid_shape": list(full_feature.shape[:3]),
                    "full_feature_shape": list(full_feature.shape),
                    "feature_shape": list(pair_feature.shape),
                    "source_latent_shape": noise_info.get("source_latent_shape"),
                    "transformer_latent_shape": noise_info.get("transformer_latent_shape"),
                    "model_output_shape": noise_info.get("model_output_shape"),
                    "source": "wan_flf2v_pair_endpoint81",
                }
                write_cache(out_path, pair_feature, cache_meta)
                written += 1
        if (local_sample_idx + 1) % max(1, int(args.log_every)) == 0:
            logging.info(
                "Processed %d/%d samples; written=%d skipped=%d",
                local_sample_idx + 1,
                len(specs),
                written,
                skipped,
            )
    logging.info("Done. written=%d skipped=%d cache_root=%s", written, skipped, cache_root)


if __name__ == "__main__":
    main()
