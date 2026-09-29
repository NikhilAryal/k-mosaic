from __future__ import annotations

__version__ = "0.2.0"

from typing import Any, Type

from .base import AttackResult, BaseAttack

ATTACKS: dict[str, Type[BaseAttack]] = {}

DEFAULT_ATTACK = "algen"


def register_attack(name: str, cls: Type[BaseAttack]) -> None:
    ATTACKS[name] = cls


def get_attack(name: str = DEFAULT_ATTACK, /, **kwargs: Any) -> BaseAttack:
    _load_builtins()
    try:
        cls = ATTACKS[name]
    except KeyError:
        raise KeyError(
            f"Unknown attack {name!r}. Known: {sorted(ATTACKS)}"
        ) from None
    return cls(**kwargs)


def attack_class(name: str = DEFAULT_ATTACK) -> Type[BaseAttack]:
    _load_builtins()
    try:
        return ATTACKS[name]
    except KeyError:
        raise KeyError(f"Unknown attack {name!r}. Known: {sorted(ATTACKS)}") from None


def available() -> dict[str, str]:
    _load_builtins()
    return {n: (c.__doc__ or "").strip().splitlines()[0] for n, c in sorted(ATTACKS.items())}


_LOADED = False


def _load_builtins() -> None:
    global _LOADED
    if _LOADED:
        return
    _LOADED = True
    from . import algen as _algen  
    from . import steer as _steer 
    from . import teia as _teia  
    from . import zero2text as _zero2text  
    from .floors import (PriorOnlyFloor, RandomVectorFloor,
                         TeiaPriorFloor, TeiaRandomFloor)

    register_attack("floor_prior", PriorOnlyFloor)
    register_attack("floor_random", RandomVectorFloor)
    register_attack("teia_floor_prior", TeiaPriorFloor)
    register_attack("teia_floor_random", TeiaRandomFloor)


__all__ = [
    "AttackResult",
    "BaseAttack",
    "ATTACKS",
    "DEFAULT_ATTACK",
    "get_attack",
    "attack_class",
    "available",
    "register_attack",
]
