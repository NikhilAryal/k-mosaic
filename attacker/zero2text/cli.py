"""The ``--z2t-*`` flags, declared by the method rather than by the shared CLI.

:class:`attacker.base.BaseAttack` puts argument-building on the method for a reason:
adding an attack should not mean editing ``train_algen.py``'s parser, which every
metrics entry point inherits. These flags are added by :mod:`metrics.ladder` through
:func:`add_attack_args` and read back by :func:`attack_kwargs`, so the flag list and
the constructor cannot drift apart.

Defaults are chosen to be runnable rather than maximal: ``Qwen2.5-0.5B-Instruct``
with 4 rounds over a 512-text seed pool costs ~4 pool embeddings plus 4 verification
passes per target set, which is minutes, not hours. The paper's setting is a larger
LLM and more rounds; ``--z2t-llm`` and ``--z2t-rounds`` are the two knobs that buy
attacker strength, and both belong on the x-axis of any plot that claims the attack
is or is not stopped by a defense.
"""

from __future__ import annotations

from typing import Any

#: Constructor kwargs this method adds on top of ALGEN's.
Z2T_KWARGS = (
    "llm", "llm_device", "llm_dtype", "llm_batch_size", "llm_temperature",
    "llm_top_p", "rounds", "seed_pool", "vary_per", "keep_best", "pool_cap",
    "local_pairs",
)


def add_attack_args(parser: Any) -> None:
    """Contribute the ``--z2t-*`` group to a parser that already has ALGEN's flags."""
    g = parser.add_argument_group("zero2text (--attack zero2text)")
    g.add_argument("--z2t-llm", dest="z2t_llm", default="Qwen/Qwen2.5-0.5B-Instruct",
                   help="candidate proposer. Nothing about it is victim-specific")
    g.add_argument("--z2t-llm-device", dest="z2t_llm_device", default=None,
                   help="put the proposer on its own GPU (default: the generator's)")
    g.add_argument("--z2t-llm-dtype", dest="z2t_llm_dtype", default="bfloat16")
    g.add_argument("--z2t-llm-batch-size", dest="z2t_llm_batch_size", type=int, default=64)
    g.add_argument("--z2t-temperature", dest="z2t_llm_temperature", type=float, default=1.0)
    g.add_argument("--z2t-top-p", dest="z2t_llm_top_p", type=float, default=0.95)
    g.add_argument("--z2t-rounds", dest="z2t_rounds", type=int, default=4,
                   help="recursive online alignment rounds. Round 0 fits the "
                        "unconditioned pool; later rounds refit on a pool that has "
                        "migrated toward the targets")
    g.add_argument("--z2t-seed-pool", dest="z2t_seed_pool", type=int, default=512,
                   help="candidates in the unconditioned round-0 pool")
    g.add_argument("--z2t-vary-per", dest="z2t_vary_per", type=int, default=2,
                   help="paraphrases generated per kept reconstruction, per round")
    g.add_argument("--z2t-keep-best", dest="z2t_keep_best", type=int, default=64,
                   help="how many distinct best-so-far reconstructions are varied")
    g.add_argument("--z2t-pool-cap", dest="z2t_pool_cap", type=int, default=1024,
                   help="maximum pairs carried into one ridge solve")
    g.add_argument("--z2t-local-pairs", dest="z2t_local_pairs", type=int, default=0,
                   help="0 = ONE map across every target, matching ALGEN and STEER. "
                        ">0 refits per target on its k nearest pool entries -- the "
                        "locally-linear attacker the F.d argument predicts, and the "
                        "only partition-aware arm in the repo. One pinv per target")


def attack_kwargs(args: Any) -> dict[str, Any]:
    """``--z2t-*`` -> :class:`~attacker.zero2text.attack.Zero2TextAttacker` kwargs."""
    return {k: getattr(args, f"z2t_{k}") for k in Z2T_KWARGS if hasattr(args, f"z2t_{k}")}
