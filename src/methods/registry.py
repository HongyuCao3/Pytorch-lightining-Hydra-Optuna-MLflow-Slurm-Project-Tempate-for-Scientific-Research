"""Method registry: maps method name -> LightningModule class.

Usage
-----
    from src.methods.registry import register, get_method

    @register("my_method")
    class MyLitModule(pl.LightningModule):
        ...

    cls = get_method("my_method")  # -> MyLitModule
"""
from __future__ import annotations

from typing import Dict, Type

import pytorch_lightning as pl

_REGISTRY: Dict[str, Type[pl.LightningModule]] = {}


def register(name: str):
    """Class decorator to register a LightningModule under *name*."""
    def decorator(cls: Type[pl.LightningModule]) -> Type[pl.LightningModule]:
        if name in _REGISTRY:
            raise KeyError(f"Method '{name}' is already registered.")
        _REGISTRY[name] = cls
        return cls
    return decorator


def get_method(name: str) -> Type[pl.LightningModule]:
    """Return the LightningModule class registered under *name*."""
    if name not in _REGISTRY:
        available = list(_REGISTRY.keys())
        raise KeyError(f"Method '{name}' not found. Available: {available}")
    return _REGISTRY[name]


def list_methods() -> list[str]:
    return list(_REGISTRY.keys())
