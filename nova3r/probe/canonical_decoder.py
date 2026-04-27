from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from omegaconf import OmegaConf

from nova3r.heads.pts3d_decoder import PointJointFMDecoderV2


def _torch_dtype_from_name(name: str) -> torch.dtype:
    name = (name or '').lower()
    if name in {'bf16', 'bfloat16'}:
        return torch.bfloat16
    if name in {'fp16', 'float16', 'half'}:
        return torch.float16
    return torch.float32


class FrozenCanonicalPointDecoder:
    """A minimal, freezeable Stage-1 NOVA3R point decoder interface.

    This wrapper intentionally loads only the Stage-1 `pts3d_head` weights from the
    NOVA3R AE checkpoint, instead of instantiating the full point-conditioned model.
    That keeps the probe interface lightweight and avoids unnecessary encoder-side
    dependencies when we only need:

        scene tokens -> complete point cloud
    """

    def __init__(
        self,
        ckpt_path: str | Path,
        device: str | torch.device = 'cuda',
        step_size: Optional[float] = None,
        sampling_method: Optional[str] = None,
        amp_dtype: Optional[str] = None,
    ) -> None:
        self.ckpt_path = Path(ckpt_path)
        if not self.ckpt_path.exists():
            raise FileNotFoundError(f'Checkpoint not found: {self.ckpt_path}')

        self.device = torch.device(device)
        self.cfg = self._load_experiment_config(self.ckpt_path)
        self.decoder_params = dict(self.cfg.model.params.cfg.pts3d_head.params)
        self.decoder_name = str(self.cfg.model.params.cfg.pts3d_head.name)
        if self.decoder_name != 'PointJointFMDecoderV2':
            raise NotImplementedError(
                f'Expected PointJointFMDecoderV2, got {self.decoder_name}. '
                'Extend FrozenCanonicalPointDecoder before using another decoder family.'
            )

        self.token_dim = int(self.decoder_params['dim_in'])
        self.num_scene_tokens = int(self.decoder_params['num_3d_tokens'])
        self.default_step_size = float(step_size or self.cfg.get('fm_step_size', 0.04))
        self.default_sampling_method = str(sampling_method or self.cfg.get('fm_sampling', 'euler'))
        self.default_amp_dtype_name = str(amp_dtype or self.cfg.get('amp_dtype', 'bf16'))
        self.default_amp_dtype = _torch_dtype_from_name(self.default_amp_dtype_name)

        self.decoder = PointJointFMDecoderV2(**self.decoder_params).to(self.device)
        self._load_decoder_weights()
        self.freeze()

    @staticmethod
    def _load_experiment_config(ckpt_path: Path):
        config_path = ckpt_path.parent / '.hydra' / 'config.yaml'
        if not config_path.exists():
            raise FileNotFoundError(
                f'Expected Hydra config next to checkpoint, but not found: {config_path}'
            )
        cfg = OmegaConf.load(config_path)
        return cfg.experiment if 'experiment' in cfg else cfg

    def _load_decoder_weights(self) -> None:
        ckpt = torch.load(self.ckpt_path, map_location='cpu')
        state_dict = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
        decoder_state = {
            key[len('pts3d_head.'):]: value
            for key, value in state_dict.items()
            if key.startswith('pts3d_head.')
        }
        missing, unexpected = self.decoder.load_state_dict(decoder_state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f'Failed to load decoder cleanly. missing={missing}, unexpected={unexpected}'
            )

    def freeze(self) -> 'FrozenCanonicalPointDecoder':
        self.decoder.eval()
        for param in self.decoder.parameters():
            param.requires_grad_(False)
        return self

    def to(self, device: str | torch.device) -> 'FrozenCanonicalPointDecoder':
        self.device = torch.device(device)
        self.decoder.to(self.device)
        return self

    def extra_repr(self) -> Dict[str, Any]:
        return {
            'checkpoint': str(self.ckpt_path),
            'num_scene_tokens': self.num_scene_tokens,
            'token_dim': self.token_dim,
            'step_size': self.default_step_size,
            'sampling_method': self.default_sampling_method,
            'amp_dtype': self.default_amp_dtype_name,
        }

    def _amp_context(self):
        if self.device.type != 'cuda':
            return nullcontext()
        return torch.amp.autocast('cuda', dtype=self.default_amp_dtype)

    @staticmethod
    def _validate_tokens(tokens: torch.Tensor, expected_tokens: int, expected_dim: int) -> torch.Tensor:
        if tokens.ndim == 2:
            tokens = tokens.unsqueeze(0)
        if tokens.ndim != 3:
            raise ValueError(f'Expected tokens with shape [B, N, D], got {tuple(tokens.shape)}')
        if tokens.shape[1] != expected_tokens:
            raise ValueError(
                f'Expected {expected_tokens} scene tokens, got {tokens.shape[1]}'
            )
        if tokens.shape[2] != expected_dim:
            raise ValueError(
                f'Expected token dim {expected_dim}, got {tokens.shape[2]}'
            )
        return tokens

    def step(self, tokens: torch.Tensor, points: torch.Tensor, t: torch.Tensor, num_views: Optional[torch.Tensor] = None) -> torch.Tensor:
        tokens = self._validate_tokens(tokens, self.num_scene_tokens, self.token_dim)
        if t.ndim == 0:
            t = torch.full((points.shape[0], points.shape[1]), float(t), device=points.device, dtype=points.dtype)
        elif t.ndim == 1:
            t = t[:, None].expand(points.shape[0], points.shape[1])
        elif t.ndim != 2:
            raise ValueError(f'Expected timestep scalar/1D/2D tensor, got shape={tuple(t.shape)}')

        with torch.no_grad():
            with self._amp_context():
                velocity = self.decoder(
                    [tokens.float()],
                    query_points=points.float(),
                    timestep=t.float(),
                    num_views=num_views,
                )
        return velocity.to(dtype=points.dtype)

    def sample_from_tokens(
        self,
        tokens: torch.Tensor,
        num_queries: int = 50000,
        x_init: Optional[torch.Tensor] = None,
        step_size: Optional[float] = None,
        num_steps: Optional[int] = None,
        seed: int = 0,
        num_views: Optional[int | torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.default_sampling_method != 'euler':
            raise NotImplementedError(
                f"This minimal interface currently implements Euler only, got {self.default_sampling_method}."
            )

        tokens = self._validate_tokens(tokens.to(self.device), self.num_scene_tokens, self.token_dim)
        batch_size = tokens.shape[0]
        step_size = float(step_size or self.default_step_size)
        num_steps = int(num_steps or max(2, int(1 // step_size)))
        time_grid = torch.linspace(0.0, 1.0, num_steps, device=self.device)

        if x_init is None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)
            x = torch.rand(batch_size, num_queries, 3, device=self.device, generator=generator) * 2 - 1
        else:
            x = x_init.to(self.device)
            if x.ndim == 2:
                x = x.unsqueeze(0)

        if isinstance(num_views, int):
            num_views = torch.full((batch_size,), float(num_views), device=self.device)
        elif isinstance(num_views, torch.Tensor):
            num_views = num_views.to(self.device)

        for idx in range(len(time_grid) - 1):
            t_cur = torch.full((batch_size, x.shape[1]), float(time_grid[idx]), device=self.device, dtype=x.dtype)
            velocity = self.step(tokens=tokens, points=x, t=t_cur, num_views=num_views)
            dt = time_grid[idx + 1] - time_grid[idx]
            x = x + dt * velocity

        return x
