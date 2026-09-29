from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

CACHE_DIR = Path(__file__).resolve().parent / "cache"


@dataclass
class ANNCorpus:
    """Clean vectors of the indexed documents plus the exact answer for each query."""

    vectors: np.ndarray                 # (N, d) float32, clean, as encoded
    ids: list[str]
    dataset: str
    model: str
    query_rows: np.ndarray              # (Q,) row indices into `vectors`
    _truth: dict[tuple[str, int], np.ndarray] = field(default_factory=dict, repr=False)

    @property
    def n(self) -> int:
        return len(self.ids)

    def truth(self, k: int, metric: str) -> np.ndarray:
        """Exact top-``k`` clean neighbours of each query row, self excluded."""
        key = (metric, k)
        if key not in self._truth:
            from .index import prepare

            X = prepare(self.vectors, metric)
            self._truth[key] = exact_topk(X, X[self.query_rows], k, exclude=self.query_rows)
        return self._truth[key]

    def describe(self) -> str:
        return (f"{self.n:,} {self.dataset} docs through {self.model}, "
                f"{len(self.query_rows):,} doc-queries")


def as_tensor(X: np.ndarray, device: str) -> torch.Tensor:
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*non-writable.*")
        return torch.as_tensor(X, device=device)


def exact_topk(
    X: np.ndarray,
    Q: np.ndarray,
    k: int,
    *,
    exclude: np.ndarray | None = None,
    chunk: int = 1024,
) -> np.ndarray:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xt = as_tensor(X, dev)
    out = np.empty((len(Q), k), dtype=np.int64)
    for s in range(0, len(Q), chunk):
        q = as_tensor(Q[s:s + chunk], dev)
        scores = q @ Xt.T
        if exclude is not None:
            rows = torch.arange(len(q), device=dev)
            cols = torch.as_tensor(exclude[s:s + chunk], device=dev)
            scores[rows, cols] = float("-inf")
        out[s:s + chunk] = scores.topk(k, dim=1).indices.cpu().numpy()
    del Xt
    return out


def load_corpus(args: Any) -> ANNCorpus:
    import train_algen as algen_pipeline
    from attacker.data import (available_records, build_splits, collect_target_ids,
                               find_embedding_sets, victim_embedder)
    from models import EmbeddingSet

    paths = algen_pipeline.victim_paths(args)
    embset = EmbeddingSet.load(paths[0])
    dataset = str(args.attack_dataset)
    holdout = collect_target_ids(find_embedding_sets("all"), dataset)
    n = int(args.ann_docs or 0) or available_records(dataset, holdout_ids=holdout)

    splits = build_splits(dataset, holdout_ids=holdout, n_train=0, n_val=n, n_align=0,
                          seed=args.seed)
    ids, texts = list(splits.val_ids), list(splits.val)
    digest = hashlib.sha1("\n".join(ids).encode()).hexdigest()[:12]
    cache = CACHE_DIR / f"{embset.model}__{dataset.replace('/', '-')}__n{len(ids)}__{digest}.npy"

    if cache.exists():
        vectors = np.load(cache, mmap_mode="r")
        print(f"[ann] corpus: cached {cache.name} (mmap)")
    else:
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        embedder = victim_embedder(embset, device=device, batch_size=512)
        print(f"[ann] encoding {len(ids):,} {dataset} docs through {embedder} "
              f"(one-time; cached to {cache})")
        vectors = np.asarray(embedder.encode(texts, show_progress=True), dtype=np.float32)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, vectors)

    n_q = min(int(args.ann_queries), len(ids))
    # Queries are the first rows of an already-shuffled sample, so they are a uniform
    # draw from the corpus and include the brute-force utility slice.
    corpus = ANNCorpus(vectors=vectors, ids=ids, dataset=dataset, model=embset.model,
                       query_rows=np.arange(n_q))
    print(f"[ann] corpus: {corpus.describe()}")
    return corpus


__all__ = ["ANNCorpus", "as_tensor", "exact_topk", "load_corpus", "CACHE_DIR"]
