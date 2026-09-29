from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from dataloader import get_dataset
from models import EmbeddingSet, get_model

DEFAULT_EMBEDDING_DIR = Path("data/embeddings")


def find_embedding_sets(
    spec: str | Sequence[str], embedding_dir: str | Path = DEFAULT_EMBEDDING_DIR
) -> list[Path]:
    embedding_dir = Path(embedding_dir)
    if isinstance(spec, str):
        if spec == "all":
            return sorted(embedding_dir.glob("*.npz"))
        if any(ch in spec for ch in "*?["):
            direct = sorted(Path().glob(spec))
            return direct or sorted(embedding_dir.glob(spec))
        spec = [spec]
    paths = [Path(s) for s in spec]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"No such embedding file(s): {missing}")
    return paths


def load_embedding_set(path: str | Path) -> EmbeddingSet:
    return EmbeddingSet.load(path)


def victim_embedder(embset: EmbeddingSet, **overrides: Any) -> Any:
    meta = embset.meta or {}
    kwargs: dict[str, Any] = {"cache_dir": None}
    for key in ("model_id", "normalize", "max_seq_length", "dtype", "prompt"):
        if key in meta:
            kwargs[key] = meta[key]
    if kwargs.get("model_id") is None:
        kwargs.pop("model_id", None)
    kwargs.update(overrides)
    return get_model(embset.model, **kwargs)


def verify_victim_encoder(
    embset: EmbeddingSet,
    embedder: Any,
    texts: Sequence[str],
    n: int = 3,
    tol: float = 1e-3,
) -> dict[str, float]:
    n = min(n, len(texts))
    if n == 0:
        return {"checked": 0, "min_cosine": float("nan")}
    fresh = embedder.encode(list(texts[:n]))
    stored = embset.vectors[:n]
    fresh_n = fresh / np.maximum(np.linalg.norm(fresh, axis=1, keepdims=True), 1e-12)
    stored_n = stored / np.maximum(np.linalg.norm(stored, axis=1, keepdims=True), 1e-12)
    cos = (fresh_n * stored_n).sum(axis=1)
    out = {"checked": n, "min_cosine": float(cos.min()), "mean_cosine": float(cos.mean())}
    status = "OK" if out["min_cosine"] > 1 - tol else "MISMATCH"
    print(
        f"reproduction check on {n} row(s): min cos={out['min_cosine']:.6f} [{status}]"
    )
    if status == "MISMATCH":
        print(
            "WARNING: the rebuilt encoder does not reproduce the stored "
            "vectors. Alignment will be fitted in the wrong space."
        )
    return out


@dataclass
class CorpusSplits:

    train: list[str] = field(default_factory=list) 
    val: list[str] = field(default_factory=list) 
    align: list[str] = field(default_factory=list)   # the few-shot known pairs
    train_ids: list[str] = field(default_factory=list)
    val_ids: list[str] = field(default_factory=list)
    align_ids: list[str] = field(default_factory=list)
    dataset: str = ""
    holdout_ids: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "train": len(self.train),
            "val": len(self.val),
            "align": len(self.align),
            "holdout": len(self.holdout_ids),
        }


def build_splits(
    dataset: str = "nfcorpus",
    *,
    holdout_ids: Iterable[str] = (),
    n_train: int = 3000,
    n_val: int = 200,
    n_align: int = 100,
    seed: int = 42,
    min_chars: int = 1,
    dataset_kwargs: dict[str, Any] | None = None,
) -> CorpusSplits:
    ds = get_dataset(dataset, min_chars=min_chars, **(dataset_kwargs or {})).load()
    holdout = set(holdout_ids)
    pool = [r for r in ds.records if r.id not in holdout]

    rng = random.Random(seed)
    order = list(range(len(pool)))
    rng.shuffle(order)

    wanted = n_align + n_val + n_train
    if wanted > len(order):
        raise ValueError(
            f"{dataset} has {len(order)} usable records after holding out "
            f"{len(holdout)} target id(s); asked for {wanted} "
            f"(align={n_align} + val={n_val} + train={n_train})."
        )

    def take(start: int, count: int) -> tuple[list[str], list[str]]:
        chunk = [pool[i] for i in order[start : start + count]]
        return [r.text for r in chunk], [r.id for r in chunk]

    align, align_ids = take(0, n_align)
    val, val_ids = take(n_align, n_val)
    train, train_ids = take(n_align + n_val, n_train)

    return CorpusSplits(
        train=train,
        val=val,
        align=align,
        train_ids=train_ids,
        val_ids=val_ids,
        align_ids=align_ids,
        dataset=dataset,
        holdout_ids=sorted(holdout),
    )


def truncate_reference(
    texts: Sequence[str], tokenizer: Any, max_length: int, device: Any = "cpu"
) -> list[str]:
    import torch

    punct_ids = tokenizer.convert_tokens_to_ids([".", "?", "!"])
    approved = punct_ids + [tokenizer.eos_token_id, tokenizer.pad_token_id]
    tokens = tokenizer(
        list(texts), padding="max_length", truncation=True,
        max_length=max_length, return_tensors="pt",
    )
    input_ids = tokens["input_ids"]
    mask = ~torch.isin(input_ids[:, -2], torch.tensor(approved))
    input_ids[mask, -2] = punct_ids[0]
    return [t.strip() for t in tokenizer.batch_decode(input_ids, skip_special_tokens=True)]


def available_records(
    dataset: str, *, holdout_ids: Iterable[str] = (), min_chars: int = 1, **dataset_kwargs: Any
) -> int:
    ds = get_dataset(dataset, min_chars=min_chars, **dataset_kwargs).load()
    holdout = set(holdout_ids)
    return sum(1 for r in ds.records if r.id not in holdout)


def texts_for_ids(dataset: str, ids: Sequence[str], **dataset_kwargs: Any) -> list[str]:
    ds = get_dataset(dataset, **dataset_kwargs).load()
    return [ds.by_id(i).text for i in ids]


def _corpus(name: str) -> str:
    try:
        return get_dataset(name).name
    except Exception:  # an unregistered name: fall back to the literal string
        return str(name)


def collect_target_ids(paths: Iterable[str | Path], dataset: str | None = None) -> list[str]:
    want = _corpus(dataset) if dataset else None
    ids: set[str] = set()
    for p in paths:
        es = EmbeddingSet.load(p)
        if want is None or _corpus(es.dataset) == want:
            ids.update(es.ids)
    return sorted(ids)
