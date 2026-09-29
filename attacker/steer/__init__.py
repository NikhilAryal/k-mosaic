from __future__ import annotations

from .. import register_attack
from .attack import DEFENSE_SCOPES, SteerAttacker

register_attack("steer", SteerAttacker)

__all__ = ["SteerAttacker", "DEFENSE_SCOPES"]
