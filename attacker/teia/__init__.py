from __future__ import annotations

from .. import register_attack
from .attack import DEFENSE_SCOPES, TeiaAttacker
from .modules import Discriminator, LinearProjection, MappingNetwork
from .trainer import TeiaTrainer

register_attack("teia", TeiaAttacker)

__all__ = [
    "TeiaAttacker",
    "TeiaTrainer",
    "MappingNetwork",
    "Discriminator",
    "LinearProjection",
    "DEFENSE_SCOPES",
]
