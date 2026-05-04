from __future__ import annotations

from typing import Any, Callable, Dict


class Registry(dict):
    def register(self, name: str) -> Callable[[Any], Any]:
        def decorator(obj: Any) -> Any:
            if name in self:
                raise KeyError(f"Duplicate registry key: {name}")
            self[name] = obj
            return obj

        return decorator


FEATURE_EXTRACTORS: Registry = Registry()
