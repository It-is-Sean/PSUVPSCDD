#!/usr/bin/env python3
"""Precompute SCRREAM pair features from a WAN-T2V video-context window.

This script is intentionally an offline bridge: it loads WAN, extracts frozen
features, and writes small per-sample cache files for adapter training. The
training loop should read these caches instead of running WAN every step.
"""

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
    """Provide torch.nn.RMSNorm on the repo's torch 2.3 runtime.

    Diffusers' WAN transformer is written against newer PyTorch releases where
    RMSNorm is in torch.nn. The nova3r environment is pinned to torch 2.3.1, so
    install a small compatible module before importing the WAN transformer.
    """
    if hasattr(torch.nn, "RMSNorm"):
        return

    class RMSNorm(torch.nn.Module):
        def __init__(
            self,
            normalized_shape,
            eps: float | None = None,
            elementwise_affine: bool = True,
            device=None,
            dtype=None,
        ) -> None:
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
    """Drop new SDPA kwargs unsupported by torch 2.3.

    Diffusers 0.37 can pass enable_gqa to scaled_dot_product_attention. The
    repo's torch 2.3.1 SDPA implementation does not expose that keyword, while
    WAN's current attention path passes it as false. Remove the kwarg and keep
    execution on CUDA.
    """
    original = torch.nn.functional.scaled_dot_product_attention
    if getattr(original, "_wan_t2v_compat", False):
        return

    def scaled_dot_product_attention_compat(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, **kwargs):
        enable_gqa = kwargs.pop("enable_gqa", False)
        scale = kwargs.pop("scale", None)
        if enable_gqa:
            raise RuntimeError("torch 2.3 scaled_dot_product_attention does not support enable_gqa=True")
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected SDPA kwargs for torch 2.3 compatibility wrapper: {unexpected}")
        return original(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
        )

    scaled_dot_product_attention_compat._wan_t2v_compat = True  # type: ignore[attr-defined]
    torch.nn.functional.scaled_dot_product_attention = scaled_dot_product_attention_compat


ensure_torch_rmsnorm_compat()
ensure_torch_sdpa_compat()


DEFAULT_ADAPTER_DATA = (
    "experiments/probe3d/adapter_data/"
    "scrream_mesh_complete_n2_adapter_seed17_tp20000_ms500000_trainplus_test.pt"
)
DEFAULT_CACHE_ROOT = "experiments/probe3d/feature_cache/scrream_wan_t2v1p3b_ctx81"
DEFAULT_MODEL_ID = "checkpoints/wan2.1/Wan2.1-T2V-1.3B-Diffusers"
DEFAULT_LAYERS = "9,14,19,24,29"
DEFAULT_TIMESTEPS = "249,499,749"
DEFAULT_LOW_NOISE_INDEX = 999
WAN_TOKENS = (21, 30, 52)


@dataclass(frozen=True)
class WindowSpec:
    sample_id: str
    scene_id: str
    sequence_id: str
    frame_ids: tuple[int, int]
    frame_paths: tuple[str, str]
    sequence_dir: str
    window_start: int
    window_end: int
    window_paths: tuple[str, ...]
    temporal_indices: tuple[int, int]


def parse_csv_ints(value: str) -> list[int]:
    items = [int(x.strip()) for x in value.split(",") if x.strip()]
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


def temporal_index_for_offset(offset: int) -> int:
    if offset < 0 or offset > 80:
        raise ValueError(f"Frame offset {offset} is outside an 81-frame WAN window")
    return 0 if offset == 0 else ((offset - 1) // 4) + 1


def build_window_spec(meta: dict[str, Any]) -> WindowSpec:
    frame_ids_raw = meta.get("frame_ids")
    frame_paths_raw = meta.get("frame_paths")
    if not frame_ids_raw or len(frame_ids_raw) != 2:
        raise ValueError(f"Sample {meta.get('sample_id', '<unknown>')} has invalid frame_ids={frame_ids_raw!r}")
    if not frame_paths_raw or len(frame_paths_raw) != 2:
        raise ValueError(f"Sample {meta.get('sample_id', '<unknown>')} has invalid frame_paths={frame_paths_raw!r}")

    frame_ids = (int(frame_ids_raw[0]), int(frame_ids_raw[1]))
    frame_paths = (str(frame_paths_raw[0]), str(frame_paths_raw[1]))
    sequence_dir = Path(frame_paths[0]).parent.parent
    rgb_map = load_rgb_paths(sequence_dir)
    min_frame = min(rgb_map)
    max_frame = max(rgb_map)
    if max_frame - min_frame + 1 < 81:
        raise ValueError(f"Sequence {sequence_dir} has fewer than 81 numbered frames")

    mid = (frame_ids[0] + frame_ids[1]) // 2
    start = mid - 40
    start = max(min_frame, min(start, max_frame - 80))
    end = start + 80
    if not (start <= frame_ids[0] <= end and start <= frame_ids[1] <= end):
        raise ValueError(
            f"Pair {frame_ids} cannot fit in 81-frame window [{start}, {end}] for {sequence_dir}"
        )
    missing = [idx for idx in range(start, end + 1) if idx not in rgb_map]
    if missing:
        raise FileNotFoundError(f"Missing RGB frames in {sequence_dir}: first missing ids {missing[:8]}")

    temporal_indices = tuple(temporal_index_for_offset(fid - start) for fid in frame_ids)
    return WindowSpec(
        sample_id=str(meta.get("sample_id") or f"{meta.get('scene_id')}/{meta.get('sequence_id')}_{frame_ids[0]:06d}_{frame_ids[1]:06d}"),
        scene_id=str(meta.get("scene_id", "")),
        sequence_id=str(meta.get("sequence_id", sequence_dir.name)),
        frame_ids=frame_ids,
        frame_paths=frame_paths,
        sequence_dir=str(sequence_dir),
        window_start=start,
        window_end=end,
        window_paths=tuple(str(rgb_map[idx]) for idx in range(start, end + 1)),
        temporal_indices=(int(temporal_indices[0]), int(temporal_indices[1])),
    )


def load_frames(paths: tuple[str, ...]) -> list[Image.Image]:
    frames: list[Image.Image] = []
    for path in paths:
        with Image.open(path) as img:
            frames.append(img.convert("RGB").copy())
    return frames


def reshape_wan_tokens(raw: torch.Tensor) -> torch.Tensor:
    t_tokens, h_tokens, w_tokens = WAN_TOKENS
    if raw.ndim != 3 or raw.shape[0] != 1:
        raise ValueError(f"Expected WAN hidden state shape [1,N,C], got {tuple(raw.shape)}")
    expected = t_tokens * h_tokens * w_tokens
    if raw.shape[1] != expected:
        raise ValueError(f"Expected {expected} WAN tokens for 480x832/81 frames, got {raw.shape[1]}")
    return raw.squeeze(0).reshape(t_tokens, h_tokens, w_tokens, raw.shape[-1]).contiguous()


class WanT2VFeatureExtractor:
    def __init__(
        self,
        model_id: str,
        prompt: str,
        device: str,
        dtype: str,
        height: int,
        width: int,
        vae_device: str,
    ) -> None:
        self.model_id = model_id
        self.prompt = prompt
        self.device = torch.device(device)
        self.vae_device = self._resolve_vae_device(vae_device)
        self.height = int(height)
        self.width = int(width)
        if dtype == "bf16":
            self.torch_dtype = torch.bfloat16
        elif dtype == "fp16":
            self.torch_dtype = torch.float16
        elif dtype == "fp32":
            self.torch_dtype = torch.float32
        else:
            raise ValueError(f"Unsupported dtype={dtype!r}")
        self.pipe = self._load_pipeline()
        with torch.no_grad():
            self.prompt_embeds, _ = self.pipe.encode_prompt(
                prompt=self.prompt,
                do_classifier_free_guidance=False,
                device=str(self.device),
            )
            self.prompt_embeds = self.prompt_embeds.to(device=self.device, dtype=self.pipe.transformer.dtype)
        if self.device.type == "cuda" and getattr(self.pipe, "text_encoder", None) is not None:
            self.pipe.text_encoder.to("cpu")
            torch.cuda.empty_cache()

    def _resolve_vae_device(self, vae_device: str) -> torch.device:
        if vae_device == "auto":
            return self.device
        return torch.device(vae_device)

    def _load_pipeline(self):
        try:
            from diffusers import WanPipeline
            from diffusers.models import AutoencoderKLWan, WanTransformer3DModel
            from diffusers.models.modeling_outputs import Transformer2DModelOutput
            from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
            from diffusers.utils import USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers
        except ImportError as exc:
            raise ImportError(
                "WAN T2V feature extraction requires diffusers, transformers, and tokenizers. "
                "Install the WAN probe dependencies before running this script."
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
                (
                    temb,
                    timestep_proj,
                    encoder_hidden_states,
                    encoder_hidden_states_image,
                ) = self.condition_embedder(timestep, encoder_hidden_states, encoder_hidden_states_image)
                timestep_proj = timestep_proj.unflatten(1, (6, -1))
                if encoder_hidden_states_image is not None:
                    encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

                all_hidden_states = {}
                output_layers_set = set(output_layers or [])
                for idx, block in enumerate(self.blocks):
                    hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)
                    if output_hidden_states or idx in output_layers_set:
                        all_hidden_states[idx] = hidden_states.detach().clone()

                shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)
                shift = shift.to(hidden_states.device)
                scale = scale.to(hidden_states.device)
                hidden_states = (
                    self.norm_out(hidden_states.float()) * (1 + scale) + shift
                ).type_as(hidden_states)
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

        class OneStepWanPipeline(WanPipeline):
            @torch.no_grad()
            def __call__(
                self,
                video,
                t,
                output_layers,
                prompt_embeds,
                generator=None,
                noise_mode="normal",
                low_noise_index=DEFAULT_LOW_NOISE_INDEX,
            ):
                device = next(self.transformer.parameters()).device
                vae_device = getattr(self, "_probe_vae_device", device)
                video_tensor = self.video_processor.preprocess_video(
                    video, height=self._probe_height, width=self._probe_width
                ).to(vae_device, dtype=torch.float32)
                self.scheduler.set_timesteps(1000, device=device)
                if noise_mode not in {"normal", "no_noise", "low_noise"}:
                    raise ValueError(f"Unsupported WAN noise_mode={noise_mode!r}")
                latents = self.vae.encode(video_tensor).latent_dist.mean
                latents_mean = torch.tensor(self.vae.config.latents_mean).view(
                    1, self.vae.config.z_dim, 1, 1, 1
                ).to(latents.device, latents.dtype)
                latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
                    1, self.vae.config.z_dim, 1, 1, 1
                ).to(latents.device, latents.dtype)
                latents = (latents - latents_mean) * latents_std
                latents = latents.to(device)
                requested_timestep_index = int(t)
                effective_timestep_index = requested_timestep_index if noise_mode == "normal" else int(low_noise_index)
                if effective_timestep_index < 0 or effective_timestep_index >= len(self.scheduler.timesteps):
                    raise ValueError(
                        f"Effective WAN timestep index must be in [0,{len(self.scheduler.timesteps) - 1}], "
                        f"got {effective_timestep_index}"
                    )
                t_idx = torch.tensor([effective_timestep_index], dtype=torch.long, device=device)
                t_input = self.scheduler.timesteps[t_idx].to(device)
                latent_noise_applied = noise_mode != "no_noise"
                if latent_noise_applied:
                    if generator is None:
                        noise = torch.randn(latents.shape, device=device, dtype=latents.dtype)
                    else:
                        noise = torch.randn(latents.shape, device=device, dtype=latents.dtype, generator=generator)
                    transformer_latents = self.scheduler.add_noise(latents, noise, t_input)
                else:
                    transformer_latents = latents
                transformer_dtype = self.transformer.dtype
                output = self.transformer(
                    hidden_states=transformer_latents.to(dtype=transformer_dtype),
                    timestep=t_input.expand(latents.shape[0]),
                    encoder_hidden_states=prompt_embeds.to(device=device, dtype=transformer_dtype),
                    return_dict=True,
                    output_layers=output_layers,
                )
                noise_info = {
                    "noise_mode": str(noise_mode),
                    "requested_timestep_index": int(requested_timestep_index),
                    "timestep_index": int(effective_timestep_index),
                    "scheduler_timestep": float(t_input.detach().float().cpu().item()),
                    "latent_noise_applied": bool(latent_noise_applied),
                    "low_noise_index": int(low_noise_index),
                }
                return output, noise_info

        transformer = TransformerWanWithFeatureOutput.from_pretrained(
            self.model_id,
            subfolder="transformer",
            torch_dtype=self.torch_dtype,
        )
        vae = AutoencoderKLWan.from_pretrained(
            self.model_id,
            subfolder="vae",
            torch_dtype=torch.float32,
        )
        pipe = OneStepWanPipeline.from_pretrained(
            self.model_id,
            transformer=transformer,
            vae=vae,
            torch_dtype=self.torch_dtype,
        )
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=3.0)
        pipe._probe_height = self.height
        pipe._probe_width = self.width
        pipe = pipe.to(self.device)
        pipe._probe_vae_device = self.vae_device
        # Keep WAN VAE on GPU in fp32. The transformer can run bf16/fp16, but
        # VAE Conv3d is more reliable in fp32 on the repo's torch 2.3 runtime.
        pipe.vae.to(self.vae_device, dtype=torch.float32)
        pipe.set_progress_bar_config(disable=True)
        pipe.transformer.eval()
        pipe.vae.eval()
        pipe.transformer.requires_grad_(False)
        pipe.vae.requires_grad_(False)
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
        noise_mode: str,
        low_noise_index: int,
    ) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
        generator = torch.Generator(device=self.device)
        generator.manual_seed(int(seed))
        (_, hidden_states), noise_info = self.pipe(
            video=frames,
            t=int(timestep),
            output_layers=[int(x) for x in layers],
            prompt_embeds=self.prompt_embeds,
            generator=generator,
            noise_mode=str(noise_mode),
            low_noise_index=int(low_noise_index),
        )
        missing = set(layers) - set(hidden_states.keys())
        if missing:
            raise RuntimeError(f"WAN did not return requested layers: {sorted(missing)}")
        features = {int(layer): reshape_wan_tokens(hidden_states[int(layer)]).detach().cpu() for layer in layers}
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
    parser.add_argument(
        "--noise_mode",
        default="normal",
        choices=("normal", "no_noise", "low_noise"),
        help=(
            "WAN latent noise mode. normal preserves Route2 behavior; no_noise feeds clean VAE latents "
            "with low-noise timestep embedding; low_noise adds scheduler noise at --low_noise_index."
        ),
    )
    parser.add_argument(
        "--low_noise_index",
        type=int,
        default=DEFAULT_LOW_NOISE_INDEX,
        help="Scheduler timestep index used by no_noise/low_noise modes. Default 999 maps to the local WAN timestep 5.",
    )
    parser.add_argument("--window_size", type=int, default=81)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--split", default=None, help="Optional split filter, e.g. train or val. Default uses all samples.")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--vae_device",
        default="cuda",
        choices=("auto", "cpu", "cuda"),
        help="Device for WAN VAE encoding. Default keeps VAE on GPU; CPU is only for debugging.",
    )
    parser.add_argument("--dtype", default="bf16", choices=("bf16", "fp16", "fp32"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--window_only", action="store_true", help="Validate windows without loading WAN or writing feature caches.")
    parser.add_argument("--log_every", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.window_size != 81:
        raise ValueError("WAN T2V route2 currently requires --window_size 81")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    timesteps = parse_csv_ints(args.timesteps)
    layers = parse_csv_ints(args.layers)
    if any(layer < 0 or layer > 29 for layer in layers):
        raise ValueError(f"WAN T2V-1.3B layer ids must be in [0,29], got {layers}")
    if any(timestep < 0 or timestep > 999 for timestep in timesteps):
        raise ValueError(f"Timesteps must index the 1000-step scheduler in [0,999], got {timesteps}")
    if args.low_noise_index < 0 or args.low_noise_index > 999:
        raise ValueError(f"--low_noise_index must be in [0,999], got {args.low_noise_index}")
    if args.noise_mode != "normal" and timesteps != [int(args.low_noise_index)]:
        raise ValueError(
            f"{args.noise_mode} caches must use --timesteps {args.low_noise_index} so cache paths match "
            "the effective WAN timestep embedding."
        )

    payload = torch.load(args.adapter_data, map_location="cpu")
    metadata = payload.get("metadata")
    splits = payload.get("splits", ["train" for _ in payload["scene_ids"]])
    if metadata is None:
        raise ValueError(f"{args.adapter_data} does not contain metadata")
    selected_indices = [idx for idx in range(len(metadata)) if args.split is None or splits[idx] == args.split]
    if args.max_samples and args.max_samples > 0:
        selected_indices = selected_indices[: args.max_samples]
    specs = [build_window_spec(metadata[idx]) for idx in selected_indices]
    logging.info("Validated %d WAN windows from %s", len(specs), args.adapter_data)

    cache_root = Path(args.output_dir)
    manifest = {
        "adapter_data": str(args.adapter_data),
        "output_dir": str(cache_root),
        "model_id": str(args.model_id),
        "prompt": args.prompt,
        "timesteps": timesteps,
        "layers": layers,
        "noise_mode": args.noise_mode,
        "low_noise_index": int(args.low_noise_index),
        "window_size": args.window_size,
        "height": args.height,
        "width": args.width,
        "seed": args.seed,
        "vae_device": args.vae_device,
        "split": args.split,
        "sample_count": len(specs),
        "feature_shape": [3120, 1536],
        "source": "wan_t2v_video_context",
    }
    cache_root.mkdir(parents=True, exist_ok=True)
    (cache_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if args.window_only:
        logging.info("window_only enabled; wrote manifest and skipped WAN loading.")
        return

    extractor: WanT2VFeatureExtractor | None = None
    written = 0
    skipped = 0
    for sample_idx, spec in enumerate(specs):
        frames: list[Image.Image] | None = None
        for timestep in timesteps:
            missing_layers = [
                layer for layer in layers
                if args.force or not cache_path(cache_root, timestep, layer, spec.sample_id).exists()
            ]
            if not missing_layers:
                skipped += len(layers)
                continue
            if extractor is None:
                logging.info("Loading WAN T2V model from %s", args.model_id)
                extractor = WanT2VFeatureExtractor(
                    model_id=str(args.model_id),
                    prompt=args.prompt,
                    device=args.device,
                    vae_device=args.vae_device,
                    dtype=args.dtype,
                    height=args.height,
                    width=args.width,
                )
            if frames is None:
                frames = load_frames(spec.window_paths)
            feature_seed = int(args.seed + sample_idx * 1000 + timestep)
            wan_features, noise_info = extractor.extract(
                frames,
                timestep=timestep,
                layers=missing_layers,
                seed=feature_seed,
                noise_mode=args.noise_mode,
                low_noise_index=int(args.low_noise_index),
            )
            for layer, full_feature in wan_features.items():
                pair_feature = full_feature[list(spec.temporal_indices)].reshape(-1, full_feature.shape[-1]).contiguous()
                if tuple(pair_feature.shape) != (3120, 1536):
                    raise ValueError(f"Expected pair feature shape [3120,1536], got {tuple(pair_feature.shape)}")
                out_path = cache_path(cache_root, timestep, layer, spec.sample_id)
                cache_meta = {
                    **asdict(spec),
                    "model_id": str(args.model_id),
                    "prompt": args.prompt,
                    "timestep": int(timestep),
                    "noise_mode": args.noise_mode,
                    "timestep_index": int(noise_info["timestep_index"]),
                    "requested_timestep_index": int(noise_info["requested_timestep_index"]),
                    "scheduler_timestep": float(noise_info["scheduler_timestep"]),
                    "latent_noise_applied": bool(noise_info["latent_noise_applied"]),
                    "low_noise_index": int(args.low_noise_index),
                    "layer": int(layer),
                    "seed": feature_seed,
                    "source": "wan_t2v_video_context",
                    "full_feature_shape": list(full_feature.shape),
                    "feature_shape": list(pair_feature.shape),
                }
                write_cache(out_path, pair_feature, cache_meta)
                written += 1
        if (sample_idx + 1) % max(1, int(args.log_every)) == 0:
            logging.info("Processed %d/%d samples; written=%d skipped=%d", sample_idx + 1, len(specs), written, skipped)
    logging.info("Done. written=%d skipped=%d cache_root=%s", written, skipped, cache_root)


if __name__ == "__main__":
    main()
