"""Five plug points: encoder, head, loss, scorer, dataset.

A registry entry is a zero-argument-friendly constructor registered under a name.
Nothing else in the pipeline is swappable, on purpose.
"""

from __future__ import annotations

from typing import Any, Callable

KINDS = ("encoder", "head", "loss", "scorer", "dataset")

_REGISTRY: dict[str, dict[str, Callable[..., Any]]] = {kind: {} for kind in KINDS}


def register(kind: str, name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    if kind not in _REGISTRY:
        raise KeyError(f"unknown registry kind {kind!r}; expected one of {KINDS}")

    def decorator(factory: Callable[..., Any]) -> Callable[..., Any]:
        if name in _REGISTRY[kind]:
            raise ValueError(f"{kind}:{name} already registered")
        _REGISTRY[kind][name] = factory
        return factory

    return decorator


def build(kind: str, name: str, **kwargs: Any) -> Any:
    try:
        factory = _REGISTRY[kind][name]
    except KeyError as exc:
        available = ", ".join(sorted(_REGISTRY[kind])) or "<none>"
        raise KeyError(f"{kind}:{name} not registered; available: {available}") from exc
    return factory(**kwargs)


def names(kind: str) -> list[str]:
    return sorted(_REGISTRY[kind])
