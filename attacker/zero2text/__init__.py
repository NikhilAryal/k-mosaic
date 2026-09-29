"""Zero2Text — training-free inversion with no leaked pairs (arXiv:2602.01757).

The fourth threat model in this repo, and the one that tests whether the ``F·d``
characterisation is exhaustive. ALGEN, STEER and TEIA all start from pairs somebody
handed the attacker. Zero2Text starts from nothing: an LLM writes text, the victim
service embeds it, and the pair that comes back is manufactured rather than leaked.

It still pays. The currency is queries, not pairs, and
:class:`~attacker.zero2text.attack.Zero2TextAttacker` reports
``queries_per_target`` next to every leakage number so the two can be compared on
one axis.

Registers ``"zero2text"`` plus its two controls::

    from attacker import get_attack
    atk = get_attack("zero2text", checkpoint_dir="attacker/outputs/.../<run>",
                     defense="sparse", sparse_checkpoint=..., rounds=4)

Driven from :mod:`metrics.ladder` with ``--attack zero2text``. Give it its own
``--ladder-out``: the ceiling and the floors are named ``undefended.json`` /
``floor_*.json`` inside a ladder directory, and ALGEN already owns those names in
the shared tree.
"""

from __future__ import annotations

from .. import register_attack
from .attack import DEFENSE_SCOPES, Zero2TextAttacker
from .floors import Zero2TextPriorFloor, Zero2TextRandomFloor

register_attack("zero2text", Zero2TextAttacker)
register_attack("zero2text_floor_prior", Zero2TextPriorFloor)
register_attack("zero2text_floor_random", Zero2TextRandomFloor)

__all__ = [
    "Zero2TextAttacker",
    "Zero2TextPriorFloor",
    "Zero2TextRandomFloor",
    "DEFENSE_SCOPES",
]
