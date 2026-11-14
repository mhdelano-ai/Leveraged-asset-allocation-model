"""Name -> factory registry, so models can be added without touching the harness.

Register a zero-argument factory that returns a fresh, unfitted
:class:`~assetforecast.returns.models.base.Forecaster`::

    from assetforecast.returns.models.registry import register

    @register("ridge")
    def _ridge():
        return Ridge(alphas=(1, 10, 100))

    # or, for a class with a no-argument constructor:
    register("my_model", MyModel)

The engine calls the factory once per refit, so the factory must return a new
object each time. Names are used as file names under ``results/forecasts``:
letters, digits, ``_`` ``-`` ``.`` only (``@`` is reserved for
derived runs such as ``ridge@lag1``).

Every module in the ``assetforecast.returns.models`` package is imported on first lookup,
so a new file ``src/assetforecast/returns/models/gbm.py`` that calls ``register`` is found
automatically. If such a module fails to import, lookups raise with the module
name rather than silently hiding the model.
"""
from __future__ import annotations

import importlib
import pkgutil
import re
from typing import Callable

from .base import Forecaster

_NAME = re.compile(r"^[A-Za-z0-9_.@\-]+$")
_REGISTRY: dict[str, Callable[[], Forecaster]] = {}
_DISCOVERED = False
_SKIP = {"base", "registry"}


def register(name: str, factory: Callable[[], Forecaster] | None = None, *, overwrite: bool = False):
    """Register ``factory`` under ``name``; usable as a decorator."""
    if not _NAME.match(name):
        raise ValueError(f"invalid model name {name!r}: use letters, digits, _ - . @")

    def deco(f):
        if name in _REGISTRY and _REGISTRY[name] is not f and not overwrite:
            raise ValueError(f"model {name!r} is already registered")
        _REGISTRY[name] = f
        return f

    return deco if factory is None else deco(factory)


def unregister(name: str) -> None:
    _REGISTRY.pop(name, None)


def _discover() -> None:
    global _DISCOVERED
    if _DISCOVERED:
        return
    _DISCOVERED = True
    from . import __path__ as pkg_path, __name__ as pkg_name
    for mod in pkgutil.iter_modules(pkg_path):
        if mod.name in _SKIP or mod.name.startswith("_"):
            continue
        try:
            importlib.import_module(f"{pkg_name}.{mod.name}")
        except Exception as e:  # surface, don't hide
            _DISCOVERED = False
            raise ImportError(f"model module assetforecast.returns.models.{mod.name} failed to import: {e}") from e


def available() -> list[str]:
    _discover()
    return sorted(_REGISTRY)


def get_factory(name: str) -> Callable[[], Forecaster]:
    _discover()
    if name not in _REGISTRY:
        raise KeyError(f"unknown model {name!r}; registered: {', '.join(sorted(_REGISTRY))}")
    return _REGISTRY[name]


def create(name: str) -> Forecaster:
    """Build a fresh model and stamp its registry name on it."""
    model = get_factory(name)()
    if not isinstance(model, Forecaster):
        raise TypeError(f"factory for {name!r} returned {type(model).__name__}, not a Forecaster")
    model.name = name
    return model
