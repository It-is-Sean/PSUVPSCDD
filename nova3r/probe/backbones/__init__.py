from .base import FrozenFeatureExtractor
from .vggt_extractor import VGGTFeatureExtractor
from .stubs import ensure_backbone_stubs_registered

ensure_backbone_stubs_registered()

__all__ = ["FrozenFeatureExtractor", "VGGTFeatureExtractor", "ensure_backbone_stubs_registered"]
