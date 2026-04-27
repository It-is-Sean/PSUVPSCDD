from __future__ import annotations

from typing import Dict, Type

from nova3r.probe.backbones.base import FrozenFeatureExtractor
from nova3r.probe.registry import FEATURE_EXTRACTORS


class _PlannedExtractor(FrozenFeatureExtractor):
    family = "planned"
    notes = "Implementation pending; this workspace only scaffolds the registry."

    def extract(self, *args, **kwargs):
        raise NotImplementedError(f"{self.name} extractor is scaffolded but not implemented yet.")


def _make_stub(name: str, family: str) -> Type[_PlannedExtractor]:
    class Stub(_PlannedExtractor):
        pass

    Stub.name = name
    Stub.family = family
    return Stub


def ensure_backbone_stubs_registered() -> Dict[str, Type[_PlannedExtractor]]:
    planned = {
        "nova3r": _make_stub("nova3r", "image_geometry"),
        "vggt": _make_stub("vggt", "image_geometry"),
        "da3": _make_stub("da3", "image_geometry"),
        "dust3r": _make_stub("dust3r", "image_geometry"),
        "mast3r": _make_stub("mast3r", "image_geometry"),
        "pi3": _make_stub("pi3", "image_geometry"),
        "wan2.1": _make_stub("wan2.1", "video"),
        "open_sora_2.0": _make_stub("open_sora_2.0", "video"),
        "cogvideox": _make_stub("cogvideox", "video"),
        "v_jepa": _make_stub("v_jepa", "video"),
    }
    for name, cls in planned.items():
        if name not in FEATURE_EXTRACTORS:
            FEATURE_EXTRACTORS.register(name)(cls)
    return planned
