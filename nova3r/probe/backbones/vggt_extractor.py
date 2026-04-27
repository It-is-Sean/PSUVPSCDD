from __future__ import annotations

import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Sequence

import torch

from nova3r.probe.backbones.base import FrozenFeatureExtractor
from nova3r.probe.protocol import FrozenRepresentation
from nova3r.probe.registry import FEATURE_EXTRACTORS


def _default_vggt_repo() -> Path:
    return Path(__file__).resolve().parents[3] / 'third_party' / 'vggt'


def _default_vggt_weights() -> Path | None:
    candidates = [
        Path('/data1/jcd_data/cache/models/vggt/VGGT-1B/model.pt'),
        Path(__file__).resolve().parents[3] / 'checkpoints' / 'vggt' / 'model.pt',
        Path(__file__).resolve().parents[3] / 'artifacts' / 'weights' / 'vggt' / 'VGGT-1B' / 'model.pt',
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _ensure_vggt_repo_on_path(repo_path: str | Path | None = None) -> Path:
    path = Path(repo_path) if repo_path is not None else _default_vggt_repo()
    if not path.exists():
        raise FileNotFoundError(
            f'VGGT repo not found at {path}. Expected vendored code under third_party/vggt or pass --vggt-repo.'
        )
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    return path


@FEATURE_EXTRACTORS.register('vggt')
class VGGTFeatureExtractor(FrozenFeatureExtractor):
    family = 'image_geometry'
    name = 'vggt'

    def __init__(
        self,
        device: str | torch.device = 'cuda',
        model_id: str = 'facebook/VGGT-1B',
        repo_path: str | Path | None = None,
        weights_path: str | Path | None = None,
        layer: str | int = 'final',
        preprocess_mode: str = 'pad',
    ) -> None:
        self.repo_path = _ensure_vggt_repo_on_path(repo_path)
        from vggt.models.vggt import VGGT

        self.device = torch.device(device)
        self.model_id = model_id
        self.weights_path = self._resolve_weights_path(weights_path, model_id)
        self.layer = layer
        self.preprocess_mode = preprocess_mode
        self.model = self._build_model(VGGT).to(self.device).eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    @staticmethod
    def _unwrap_state_dict(payload):
        if isinstance(payload, dict):
            for key in ('state_dict', 'model'):
                value = payload.get(key)
                if isinstance(value, dict):
                    return value
        return payload

    @classmethod
    def _resolve_weights_path(cls, weights_path: str | Path | None, model_id: str) -> Path | None:
        candidates: list[Path] = []
        if weights_path is not None:
            candidates.append(Path(weights_path))

        model_path = Path(str(model_id)).expanduser()
        if model_path.exists():
            candidates.append(model_path)
            if model_path.is_dir():
                candidates.extend([
                    model_path / 'model.pt',
                    model_path / 'pytorch_model.bin',
                ])

        default_weights = _default_vggt_weights()
        if default_weights is not None:
            candidates.append(default_weights)

        for candidate in candidates:
            if candidate.exists():
                if candidate.is_dir():
                    for leaf in (candidate / 'model.pt', candidate / 'pytorch_model.bin'):
                        if leaf.exists():
                            return leaf.resolve()
                else:
                    return candidate.resolve()
        return None

    def _build_model(self, VGGT):
        if self.weights_path is None:
            return VGGT.from_pretrained(self.model_id)

        model = VGGT()
        payload = torch.load(self.weights_path, map_location='cpu')
        state_dict = self._unwrap_state_dict(payload)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f'Failed to load VGGT weights cleanly from {self.weights_path}. '
                f'missing={missing}, unexpected={unexpected}'
            )
        return model

    def _autocast_context(self):
        if self.device.type != 'cuda':
            return nullcontext()
        major, _ = torch.cuda.get_device_capability(self.device)
        dtype = torch.bfloat16 if major >= 8 else torch.float16
        return torch.cuda.amp.autocast(dtype=dtype)

    def _select_layer(self, tokens_per_layer: list[torch.Tensor]) -> tuple[torch.Tensor, int]:
        num_layers = len(tokens_per_layer)
        if isinstance(self.layer, int):
            idx = self.layer
        else:
            key = str(self.layer).lower()
            if key in {'final', 'last'}:
                idx = num_layers - 1
            elif key == 'mid':
                idx = num_layers // 2
            elif key == 'early':
                idx = 0
            else:
                raise ValueError(f'Unsupported layer selector: {self.layer}')
        idx = idx % num_layers
        return tokens_per_layer[idx], idx

    @torch.no_grad()
    def extract(self, image_paths: Sequence[str]) -> FrozenRepresentation:
        from vggt.utils.load_fn import load_and_preprocess_images

        images = load_and_preprocess_images(list(image_paths), mode=self.preprocess_mode).to(self.device)
        if images.ndim == 4:
            images = images.unsqueeze(0)

        with self._autocast_context():
            aggregated_tokens_list, patch_start_idx = self.model.aggregator(images)

        selected_tokens, layer_index = self._select_layer(aggregated_tokens_list)
        b, s, p, c = selected_tokens.shape
        flat_tokens = selected_tokens.reshape(b, s * p, c).float()
        return FrozenRepresentation(
            tokens=flat_tokens,
            meta={
                'source': 'VGGT',
                'model_id': self.model_id,
                'weights_path': str(self.weights_path) if self.weights_path is not None else None,
                'layer': self.layer,
                'layer_index': layer_index,
                'num_layers': len(aggregated_tokens_list),
                'patch_start_idx': int(patch_start_idx),
                'raw_shape': [int(b), int(s), int(p), int(c)],
                'preprocess_mode': self.preprocess_mode,
            },
        )
