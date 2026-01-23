from __future__ import annotations

from typing import Callable

from .base import Engine


_ENGINE_FACTORIES: dict[str, Callable[[], Engine]] = {}


def register_engine(name: str, factory: Callable[[], Engine]) -> None:
    key = str(name).strip().lower()
    if not key:
        raise ValueError("Engine name must be non-empty.")
    _ENGINE_FACTORIES[key] = factory


def get_engine(name: str) -> Engine:
    key = str(name).strip().lower()
    if key not in _ENGINE_FACTORIES:
        available = ", ".join(sorted(_ENGINE_FACTORIES)) or "none"
        raise KeyError(f"Engine '{name}' not registered (available: {available}).")
    return _ENGINE_FACTORIES[key]()


def list_engines() -> list[str]:
    return sorted(_ENGINE_FACTORIES)


__all__ = ["get_engine", "list_engines", "register_engine"]
