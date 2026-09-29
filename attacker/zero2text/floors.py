"""Zero2Text's two control arms.

The ALGEN floors in :mod:`attacker.floors` cannot be reused verbatim. They are
correct about the *decoder* — Zero2Text decodes through the same ``G``, so the
prior-only number would in principle transfer — but both of them build their
control out of the attacker's **corpus slice**, and Zero2Text has no corpus. A floor
that assumes in-domain data is not the floor for a method whose whole claim is that
it needs none.

So both are rebuilt over the manufactured pool:

``zero2text_floor_prior``
    ``G`` decoding one constant vector: the mean, in ``G``'s own space, of the
    **LLM seed pool**. No victim vector and no victim query is consumed, so whatever
    it scores is the LLM prior plus the decoder's prior and nothing else. This is the
    matched-prior control the SPARSE-style literature never reports.

``zero2text_floor_random``
    The full recursive loop with the pool's pairing destroyed before every ridge
    solve. The map is then refit each round on noise, so the targets are projected
    through something that cannot carry information about them — while the LLM, the
    rounds, the verification step and the query budget all stay identical.

Both reach the real attack by inheritance rather than by copy, so a change to the
loop cannot leave its own controls behind.

Research use only.
"""

from __future__ import annotations

from typing import Any

import torch

from models import EmbeddingSet

from ..floors import PRIORS, PriorOnlyFloor, RandomVectorFloor
from .attack import Zero2TextAttacker


class Zero2TextPriorFloor(PriorOnlyFloor, Zero2TextAttacker):
    """Floor: ``G`` decoding the LLM pool's mean — no target, no victim query."""

    name = "zero2text_floor_prior"

    def _prior(self, embset: EmbeddingSet) -> torch.Tensor:
        """The constant vector, built from text the attacker invented.

        ``PriorOnlyFloor._prior`` averages the corpus slice the attacker is assumed
        to own. Zero2Text owns none, so the honest analogue is the LLM's own output:
        the same seed pool the real attack starts from, embedded with ``G``'s encoder
        and renormalised. Nothing here touches the victim.
        """
        if self.prior == "zero":
            return torch.zeros(1, self.model.embedder_dim, device=self.device)
        pool = self.proposer.seed_pool(self.seed_pool)
        if not pool:
            raise SystemExit(
                "zero2text_floor_prior: the LLM produced no usable seed text, so the "
                "prior cannot be built. Check --z2t-llm."
            )
        Y = self._target_embeddings(pool)
        mean = Y.mean(dim=0, keepdim=True)
        print(f"[zero2text_floor_prior] prior from {len(pool)} LLM-generated text(s), "
              f"0 victim queries")
        return mean / mean.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)

    def invert(self, embset: EmbeddingSet, source: str = "") -> Any:
        result = super().invert(embset, source=source)
        result.attack = self.name
        result.config.update({
            "arm": "floor_prior_only",
            "threat_model": "zero2text control - LLM prior only",
            "llm": self.llm_id,
            "prior_note": f"{PRIORS[self.prior]} (rebuilt over the LLM pool, not a corpus)",
            "align_samples": 0,
        })
        result.diagnostics["victim_queries"] = 0
        return result


class Zero2TextRandomFloor(RandomVectorFloor, Zero2TextAttacker):
    """Floor: the full recursive loop, with the manufactured pool's pairing destroyed.

    ``_corrupt_pairs`` is inherited unchanged from :class:`RandomVectorFloor` and is
    invoked by :meth:`Zero2TextAttacker._solve` on every round, so the corruption is
    applied *after* the defense and *before* each solve — the same ordering the
    ALGEN floor documents, repeated once per round because the map is.

    Note what this controls for and what it does not: the verification step still
    re-embeds the reconstructions and keeps the best, so a candidate that happens to
    land near a target is still retained. That is deliberate. It is the LLM-prior
    contribution that survives when the *map* carries nothing, which is precisely the
    quantity a training-free attack's leakage number has to be discounted by.
    """

    name = "zero2text_floor_random"


__all__ = ["Zero2TextPriorFloor", "Zero2TextRandomFloor"]
