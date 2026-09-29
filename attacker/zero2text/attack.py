"""Zero2Text: recursive online alignment with no leaked pairs (arXiv:2602.01757).

Kim et al. describe a *training-free* inversion attack: an LLM proposes text, a
"dynamic ridge regression" maps the victim's space into the attacker's on the fly,
and the two iterate until the generated text's embedding sits on the target. No
decoder is trained per encoder, no in-domain corpus is assumed, and the paper
reports that DP-style perturbation does not stop it.

Why it belongs in this repo, stated plainly, because it is the reason to spend the
GPU hours: the ``F·d`` characterisation says a recall-preserving transform's only
security contribution is the number of leaked pairs an attacker must fit. Zero2Text
looks like the counterexample — it leaks nothing. It is not one. It **manufactures**
its pairs by querying the victim service, so the bound survives with pairs replaced
by queries, and the interesting quantity becomes *how many queries* each defense
costs the attacker. This class reports that number.

Threat model, against the other three:

===================  =======================  =======================  =======================
                     ALGEN                    TEIA                     Zero2Text (here)
===================  =======================  =======================  =======================
leaked pairs         ``k`` from a corpus      ``D_L`` from a breach    **none**
the attacker owns    (``--align-samples``)    (``--leaked-samples``)
query access         yes, budgeted            **no**                   yes — and it is the
                                                                       whole attack
in-domain corpus     yes (also trains ``G``)  external only            **none**; the pool is
                                                                       LLM-generated
the map              one ridge solve, once    none (trained adapter)   refit every round, on
                                                                       a pool that migrates
                                                                       toward the targets
===================  =======================  =======================  =======================

Everything downstream of the map is ALGEN's — the same generator ``G``, the same
decode, the same references, the same truncation — because holding those fixed is
what makes the four attacks comparable. The only thing that changes is where the
pairs come from, which is exactly the variable under study.

**What is faithful and what is a stand-in.** The recursive loop, the online ridge
solve, the LLM prior and the verification-by-re-embedding are the paper's. The
decode step is ``G`` rather than the paper's own generator, for the comparability
reason above; consequently the ceiling here is ``G``'s ceiling, and cross-attack
comparison goes through ``leak_norm`` — normalised by this attack's own floor and
ceiling — never through raw token F1.

**Cost.** One round costs ``|pool|`` encoder queries for the pairs plus ``n``
queries to verify the current reconstructions. ``queries_per_target`` in the
diagnostics is the headline: it is the axis on which a query budget, rather than a
noise budget, would be the defense.

Registered as ``"zero2text"``::

    python -m metrics.ladder --attack zero2text --ladder-out metrics/outputs/zero2text ...

Research use only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from models import EmbeddingSet

from ..algen.align import LinearAligner
from ..algen.attack import ALGENAttacker
from ..algen.utils import add_punctuation_token_ids, check_normalization
from ..base import AttackResult
from ..data import texts_for_ids, verify_victim_encoder, victim_embedder
from ..metrics import eval_embeddings, eval_texts
from .proposer import LLMProposer

#: Where the defense is allowed to touch the attack. Same vocabulary as
#: :mod:`attacker.steer.attack` and :mod:`attacker.teia.attack`.
DEFENSE_SCOPES = {
    "both": "Zero2Text's own model — the attacker queries the DEPLOYED service, so "
            "its manufactured pairs come back defended and the online ridge absorbs "
            "the defense. This is the adaptive adversary.",
    "targets": "the defense is applied storage-side and is NOT query-reachable, so "
               "the attacker's own queries return clean vectors and it must invert a "
               "transform it can never sample. The handicapped arm.",
}


def _rowwise_cos(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return (F.normalize(A, p=2, dim=1) * F.normalize(B, p=2, dim=1)).sum(dim=1)


class Zero2TextAttacker(ALGENAttacker):
    """Training-free inversion: an LLM manufactures the pairs, a ridge map is refit each round."""

    name = "zero2text"
    requires_training = True          # G still has to exist; nothing else does

    def __init__(
        self,
        *args: Any,
        defense_scope: str = "both",
        llm: str = "Qwen/Qwen2.5-0.5B-Instruct",
        llm_device: str | None = None,
        llm_dtype: str = "bfloat16",
        llm_batch_size: int = 64,
        llm_temperature: float = 1.0,
        llm_top_p: float = 0.95,
        rounds: int = 4,
        seed_pool: int = 512,
        vary_per: int = 2,
        keep_best: int = 64,
        pool_cap: int = 1024,
        local_pairs: int = 0,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            defense_scope: See :data:`DEFENSE_SCOPES`. ``both`` is Zero2Text's own
                reading and the default; ``targets`` models a storage-side transform
                the attacker's query access cannot reach.
            llm:           The candidate proposer. Nothing about it is victim-specific
                — that is the "zero-training, cross-domain" claim.
            rounds:        Recursive alignment rounds. Round 0 fits on the
                unconditioned seed pool; every later round refits on a pool that has
                migrated toward the targets.
            seed_pool:     Candidates in the unconditioned round-0 pool.
            vary_per:      Paraphrases generated per kept candidate, per round.
            keep_best:     How many distinct best-so-far reconstructions are varied.
                Capped below ``n`` targets on purpose: the pool is shared, so the
                attack pays one query per pool entry, not one per target.
            pool_cap:      Maximum pool size carried into a ridge solve. The solve is
                ``O(d^2)`` in the pairs, and an unbounded pool would make later rounds
                dominate the cost without improving the local fit.
            local_pairs:   ``0`` (default) fits ONE map across every target, matching
                what ALGEN and STEER do, so the four attacks differ in one variable.
                ``>0`` refits per target on its ``local_pairs`` nearest pool entries —
                the piecewise-linear attacker the ``F·d`` argument predicts, and the
                only arm here that is partition-aware. Expensive: one pinv per target.

            Every other argument is :class:`~attacker.algen.attack.ALGENAttacker`'s.
            ``align_samples`` is accepted and **unused**: this attacker draws no pairs
            from the corpus. It still appears in the ladder's cache-directory name, so
            two PAIRS values produce byte-identical zero2text arms in two directories.
        """
        if defense_scope not in DEFENSE_SCOPES:
            raise ValueError(
                f"defense_scope must be one of {sorted(DEFENSE_SCOPES)}, got {defense_scope!r}"
            )
        super().__init__(*args, **kwargs)
        self.defense_scope = defense_scope
        self.llm_id = llm
        self.llm_device = llm_device
        self.llm_dtype = llm_dtype
        self.llm_batch_size = llm_batch_size
        self.llm_temperature = llm_temperature
        self.llm_top_p = llm_top_p
        self.rounds = max(1, int(rounds))
        self.seed_pool = int(seed_pool)
        self.vary_per = int(vary_per)
        self.keep_best = int(keep_best)
        self.pool_cap = int(pool_cap)
        self.local_pairs = int(local_pairs)

        self._proposer: LLMProposer | None = None
        self._queries = 0
        self._vec_cache: dict[str, torch.Tensor] = {}
        self._query_cells: Any = None
        self._trace: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    # The LLM prior
    # ------------------------------------------------------------------ #

    @property
    def proposer(self) -> LLMProposer:
        if self._proposer is None:
            self._proposer = LLMProposer(
                self.llm_id,
                device=self.llm_device or self.device,
                max_new_tokens=self.max_length,
                temperature=self.llm_temperature,
                top_p=self.llm_top_p,
                batch_size=self.llm_batch_size,
                seed=self.seed,
                dtype=self.llm_dtype,
            )
        return self._proposer

    # ------------------------------------------------------------------ #
    # Query access — the resource this attack spends instead of leaked pairs
    # ------------------------------------------------------------------ #

    def _query(self, embedder: Any, texts: Sequence[str]) -> torch.Tensor:
        """Push attacker-chosen text through the victim service and count the cost.

        Under ``defense_scope="both"`` the service returns what it stores, so the
        pair is defended and passes through the index's storage layer exactly as a
        target does. Under ``"targets"`` the defense is storage-side and unreachable
        by query, so the attacker gets the clean encoder output and has to invert a
        transform it can never sample — which is the point of that arm.

        A text is queried ONCE. The attacker keeps what it gets back, so asking again
        would be a second charge for a vector it already holds. It also fixes the
        threat model for a stochastic defense: one stored draw per text, which is what
        an encode-time perturbing service actually returns. And it is where the pool's
        cell assignments are captured -- without them ``partition`` diagnostics are
        empty and a partitioned arm cannot be interpreted at all.
        """
        missing = [t for t in dict.fromkeys(texts) if t not in self._vec_cache]
        if missing:
            clean = [t if t.strip() else " " for t in missing]
            inputs = clean if self.align_text == "full" else self._truncate(clean)
            X = torch.tensor(
                embedder.encode(inputs), dtype=torch.float32, device=self.device
            )
            self._queries += len(inputs)
            if self.defense_scope == "both":
                if self.defense != "none":
                    X, self._defense_state = self._defend(X)
                X = self._stored(X)
                cells = getattr(self.storage, "last_cells", None)
                if cells is not None and self._query_cells is None:
                    self._query_cells = cells
            for t, row in zip(missing, X):
                self._vec_cache[t] = row
        return torch.stack([self._vec_cache[t] for t in texts])

    # ------------------------------------------------------------------ #
    # The online ridge solve
    # ------------------------------------------------------------------ #

    def _select(
        self, texts: list[str], X_pool: torch.Tensor, Y_pool: torch.Tensor,
        X_targets: torch.Tensor,
    ) -> tuple[list[str], torch.Tensor, torch.Tensor]:
        """Keep the ``pool_cap`` pool entries closest to ANY target.

        This is the "recursive" half of recursive online alignment, and dropping it
        silently turns the method into a one-shot fit on an LLM prior. Truncating the
        pool by recency instead -- which is the obvious implementation and the wrong
        one -- leaves each round's handful of target-adjacent paraphrases outnumbered
        by the generic seed pool they were meant to replace, so the pool never
        migrates: ``align_cos`` sits flat while the number of targets that improve
        decays round over round.

        Selecting by fitness is what makes the map LOCAL to the targets, which is the
        regime the F.d argument says a recall-preserving transform must admit.
        """
        if len(texts) <= self.pool_cap:
            return texts, X_pool, Y_pool
        fit = (F.normalize(X_pool, p=2, dim=1) @ F.normalize(X_targets, p=2, dim=1).T)
        keep = fit.max(dim=1).values.topk(self.pool_cap).indices
        print(f"[z2t] pool {len(texts)} -> {self.pool_cap} by max cos to any target "
              f"(kept mean {fit.max(dim=1).values[keep].mean():.4f})")
        return [texts[i] for i in keep.tolist()], X_pool[keep], Y_pool[keep]

    def _solve(
        self, X_pool: torch.Tensor, Y_pool: torch.Tensor, X_targets: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Fit the map on this round's pool and project the targets through it.

        One global map by default. ``--z2t-local-pairs k`` instead refits per target
        on its ``k`` nearest pool entries, which is the locally-linear attacker the
        ``F·d`` argument says a recall-preserving transform must admit.
        """
        X_pool, Y_pool = self._corrupt_pairs(X_pool, Y_pool)   # floor hook, identity here
        if self.local_pairs <= 0:
            aligner = LinearAligner(self.reg_lambda).fit(X_pool, Y_pool)
            self._aligner = aligner
            return aligner.transform(X_targets), dict(aligner.report.to_dict())

        k = min(self.local_pairs, X_pool.shape[0])
        sim = F.normalize(X_targets, p=2, dim=1) @ F.normalize(X_pool, p=2, dim=1).T
        idx = sim.topk(k, dim=1).indices
        rows, cos = [], []
        for i in range(X_targets.shape[0]):
            sel = idx[i]
            a = LinearAligner(self.reg_lambda).fit(X_pool[sel], Y_pool[sel])
            rows.append(a.transform(X_targets[i : i + 1]))
            cos.append(a.report.train_cos)
            if i == 0:
                self._aligner = a          # one representative map, for save_artifacts
        return torch.cat(rows, dim=0), {
            "X_Y_COS": float(np.mean(cos)), "n_pairs": k, "local_pairs": k,
            "reg_lambda": self.reg_lambda or 0,
            "source_dim": int(X_pool.shape[1]), "target_dim": int(Y_pool.shape[1]),
        }

    # ------------------------------------------------------------------ #
    # Decode
    # ------------------------------------------------------------------ #

    def _decode(self, hidden: torch.Tensor, refs: Sequence[str] | None = None) -> list[str]:
        """``G``, with ALGEN's decode settings — identical to the other arms."""
        if self.attention_mask == "oracle" and refs is not None:
            mask = add_punctuation_token_ids(
                list(refs), self.tokenizer, self.max_length, self.device
            )["attention_mask"]
        else:
            mask = torch.ones(
                hidden.size(0), self.max_length, dtype=torch.long, device=self.device
            )
        self.model.eval()
        generated = self.model.generate({"hidden_states": hidden, "attention_mask": mask})
        return [
            t.strip()
            for t in self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        ]

    # ------------------------------------------------------------------ #
    # BaseAttack hooks
    # ------------------------------------------------------------------ #

    def fit(self, embset: EmbeddingSet) -> None:
        """Nothing to fit ahead of time — the map is a function of the targets.

        ALGEN solves once, before it ever looks at a target. Zero2Text cannot: the
        pool migrates toward the targets, so the map only exists inside
        :meth:`invert`. This resets the per-file state instead.
        """
        self._aligner = None
        self._defense_state = {}
        self._queries = 0
        self._vec_cache = {}
        self._query_cells = None
        self._trace = []

    def invert(self, embset: EmbeddingSet, source: str = "") -> AttackResult:
        """Recursive online alignment over every row of ``embset``."""
        dataset = self.dataset_for(embset)
        full_refs = texts_for_ids(dataset, embset.ids)
        refs = self._truncate(full_refs)
        n = len(embset.ids)

        embedder = victim_embedder(embset)
        print(f"[victim] {embedder}")
        if self.verify:
            verify_victim_encoder(embset, embedder, full_refs)

        # Targets, as the index holds them.
        X = torch.tensor(embset.vectors, dtype=torch.float32, device=self.device)
        if self.defense != "none":
            X, self._defense_state = self._defend(X)
        X = self._stored(X)
        target_cells = getattr(self.storage, "last_cells", None)
        check_normalization(X, "X target (victim)")

        print(f"[z2t] {self.rounds} round(s), scope={self.defense_scope!r}, "
              f"seed_pool={self.seed_pool}, vary_per={self.vary_per}, "
              f"keep_best={self.keep_best}, local_pairs={self.local_pairs or 'off (one map)'}")

        new_texts: list[str] = self.proposer.seed_pool(self.seed_pool)
        pool_texts: list[str] = []
        pool_X: torch.Tensor | None = None
        pool_Y: torch.Tensor | None = None
        best_text = [""] * n
        best_cos = torch.full((n,), -2.0, device=self.device)
        best_hidden = torch.zeros(n, self.model.embedder_dim, device=self.device)
        pair_cells = None
        align_metrics: dict[str, Any] = {}

        for r in range(self.rounds):
            if new_texts:
                # Only the texts this round actually added reach the encoder; the rest
                # are already in the cache and were paid for when they were proposed.
                X_new = self._query(embedder, new_texts)
                Y_new = self._target_embeddings(new_texts)
                pool_texts += new_texts
                pool_X = X_new if pool_X is None else torch.cat([pool_X, X_new])
                pool_Y = Y_new if pool_Y is None else torch.cat([pool_Y, Y_new])
                if pair_cells is None:
                    pair_cells = self._query_cells
            if not pool_texts:
                print(f"[z2t] round {r}: empty pool, stopping early")
                break
            pool_texts, pool_X, pool_Y = self._select(pool_texts, pool_X, pool_Y, X)
            pool = pool_texts

            Y_hat, align_metrics = self._solve(pool_X, pool_Y, X)
            preds = self._decode(Y_hat, full_refs)

            # Online verification: re-embed the reconstruction through the SAME access
            # path and keep it only if it moved closer to the target. This is the loop's
            # only feedback signal, and the reason noise defenses degrade it gracefully
            # rather than breaking it -- a wrong candidate is rejected, not absorbed.
            X_pred = self._query(embedder, preds)
            cos = _rowwise_cos(X_pred, X)
            better = cos > best_cos
            for i in torch.nonzero(better, as_tuple=False).flatten().tolist():
                best_text[i] = preds[i]
            best_hidden[better] = Y_hat[better]
            best_cos = torch.maximum(best_cos, cos)

            self._trace.append({
                "round": r, "pool": len(pool), "improved": int(better.sum()),
                "round_cos": float(cos.mean()), "best_cos": float(best_cos.mean()),
                "align_cos": float(align_metrics.get("X_Y_COS", float("nan"))),
                "queries": self._queries,
            })
            print(f"[z2t] round {r}: pool={len(pool)} align_cos="
                  f"{align_metrics.get('X_Y_COS', float('nan')):.4f} "
                  f"verify_cos={float(cos.mean()):.4f} best={float(best_cos.mean()):.4f} "
                  f"improved={int(better.sum())}/{n} queries={self._queries}")

            if r + 1 < self.rounds:
                # Distinct best-so-far only: varying 128 near-copies of one sentence
                # buys nothing and costs a query each.
                seeds = list(dict.fromkeys(t for t in best_text if t))[: self.keep_best]
                new_texts = [
                    t for t in dict.fromkeys(self.proposer.vary(seeds, self.vary_per))
                    if t not in self._vec_cache
                ]

        predictions = [t if t else " " for t in best_text]

        # The generator's own ceiling, on the true target embedding. Same decoder, so
        # it is directly comparable to ALGEN's and STEER's oracle.
        Y_true = self._target_embeddings(full_refs)
        oracle_predictions = self._decode(Y_true, full_refs)
        test_cos, test_mse = eval_embeddings(best_hidden, Y_true)

        diagnostics = dict(align_metrics)
        diagnostics.update({
            "X_Y_test_COS": float(test_cos),
            "X_Y_test_MSEloss": float(test_mse),
            "verify_COS": float(best_cos.mean()),
            "rounds_run": len(self._trace),
            "victim_queries": int(self._queries),
            "queries_per_target": float(self._queries / max(n, 1)),
            "leaked_pairs": 0,
            "self_generated_pairs": int(len(self._vec_cache)),
            "final_pool": int(len(pool_texts)),
            "trace": self._trace,
        })
        part = self._partition_diagnostics_from(pair_cells, target_cells, X.shape[1])
        if part:
            diagnostics["partition"] = part

        return AttackResult(
            attack=self.name,
            source=source or embset.model,
            ids=list(embset.ids),
            predictions=predictions,
            references=refs,
            full_references=full_refs,
            text_metrics=eval_texts(predictions, refs),
            text_metrics_full=eval_texts(predictions, full_refs),
            oracle_predictions=oracle_predictions,
            oracle_metrics=eval_texts(oracle_predictions, refs),
            diagnostics=diagnostics,
            config={
                "checkpoint": str(self.checkpoint_dir),
                "threat_model": "zero2text (arXiv:2602.01757) - training-free, no leaked pairs",
                "generator": self.trainer.args["model_name"],
                "llm": self.llm_id,
                "rounds": self.rounds,
                "seed_pool": self.seed_pool,
                "vary_per": self.vary_per,
                "keep_best": self.keep_best,
                "pool_cap": self.pool_cap,
                "local_pairs": self.local_pairs,
                "victim_model": embset.model,
                "victim_dataset": embset.dataset,
                "dataset": dataset,
                "align_samples": 0,
                "align_samples_note": "zero2text draws NO pairs from the corpus; the "
                                      "ladder's k<N> directory name is inherited from "
                                      "ALGEN and does not describe this attack",
                "align_text": self.align_text,
                "attention_mask": self.attention_mask,
                "reg_lambda": self.reg_lambda,
                "max_length": self.max_length,
                "defense": self.defense,
                "defense_scope": self.defense_scope,
                "defense_scope_note": DEFENSE_SCOPES[self.defense_scope],
                "epsilon": self.epsilon,
                "noise_level": self.noise_level,
                "storage": self.storage.describe() if self.storage is not None else None,
                "seed": self.seed,
            },
        )

    # ------------------------------------------------------------------ #

    def _partition_diagnostics_from(
        self, pair_cells: Any, target_cells: Any, dim: int
    ) -> dict[str, Any]:
        """``ALGENAttacker._partition_diagnostics``, over the manufactured pool."""
        saved = getattr(self, "_pair_cells", None)
        self._pair_cells = pair_cells
        try:
            return self._partition_diagnostics(target_cells, dim)
        finally:
            self._pair_cells = saved

    def summary_row(self, result: AttackResult) -> dict[str, Any]:
        return {
            "align_cos": result.diagnostics.get("X_Y_test_COS", float("nan")),
            "verify_cos": result.diagnostics.get("verify_COS", float("nan")),
            "q_per_target": result.diagnostics.get("queries_per_target", float("nan")),
        }

    def save_artifacts(self, out_dir: Path) -> None:
        super().save_artifacts(out_dir)
        import json

        (out_dir / "zero2text_trace.json").write_text(json.dumps(self._trace, indent=2))
