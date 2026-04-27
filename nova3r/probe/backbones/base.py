from __future__ import annotations

from abc import ABC, abstractmethod

from nova3r.probe.protocol import FrozenRepresentation


class FrozenFeatureExtractor(ABC):
    family: str = "unknown"
    name: str = "unknown"

    @abstractmethod
    def extract(self, *args, **kwargs) -> FrozenRepresentation:
        raise NotImplementedError
