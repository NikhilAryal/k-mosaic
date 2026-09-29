from __future__ import annotations

import json
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


@dataclass(frozen=True)
class Record:
    id: str
    text: str
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Selection:
    dataset: str
    indices: list[int]
    records: list[Record]

    @property
    def ids(self) -> list[str]:
        return [r.id for r in self.records]

    @property
    def texts(self) -> list[str]:
        return [r.text for r in self.records]

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[Record]:
        return iter(self.records)


class BaseDataset(ABC):
    name: str = "base"

    def __init__(
        self,
        *,
        max_records: int | None = None,
        min_chars: int = 1,
        seed: int = 0,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.max_records = max_records
        self.min_chars = min_chars
        self.seed = seed
        self.cache_dir = Path(cache_dir) if cache_dir is not None else Path("data")
        self._records: list[Record] | None = None
        self._id_to_index: dict[str, int] | None = None

    @abstractmethod
    def _load(self) -> Iterable[Record]:
        """Produce the corpus records in a deterministic order."""

    def load(self) -> "BaseDataset":
        if self._records is not None:
            return self

        records: list[Record] = []
        for rec in self._load():
            if len(rec.text.strip()) < self.min_chars:
                continue
            records.append(rec)
            if self.max_records is not None and len(records) >= self.max_records:
                break

        self._records = records
        self._id_to_index = {r.id: i for i, r in enumerate(records)}
        if len(self._id_to_index) != len(records):
            # Non-fatal: id lookup will resolve to the first occurrence.
            pass
        return self

    @property
    def records(self) -> list[Record]:
        if self._records is None:
            self.load()
        return self._records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, key: int | slice | str) -> Record | list[Record]:
        if isinstance(key, str):
            return self.by_id(key)
        if isinstance(key, slice):
            return self.records[key]
        return self.records[key]

    def __iter__(self) -> Iterator[Record]:
        return iter(self.records)

    def __repr__(self) -> str:
        n = len(self._records) if self._records is not None else "unloaded"
        return f"{type(self).__name__}(name={self.name!r}, size={n})"

    def by_id(self, record_id: str) -> Record:
        self.load()
        idx = self._id_to_index.get(record_id)  # type: ignore[union-attr]
        if idx is None:
            raise KeyError(f"{self.name}: no record with id {record_id!r}")
        return self.records[idx]

    def index_of(self, record_id: str) -> int:
        self.load()
        idx = self._id_to_index.get(record_id)  # type: ignore[union-attr]
        if idx is None:
            raise KeyError(f"{self.name}: no record with id {record_id!r}")
        return idx

    def select(
        self,
        *,
        n: int | None = None,
        indices: Sequence[int] | None = None,
        ids: Sequence[str] | None = None,
        contains: str | None = None,
        seed: int | None = None,
    ) -> Selection:
        self.load()

        if indices is not None:
            pool = list(indices)
            for i in pool:
                if not 0 <= i < len(self):
                    raise IndexError(f"{self.name}: index {i} out of range (size {len(self)})")
        elif ids is not None:
            pool = [self.index_of(r) for r in ids]
        elif contains is not None:
            needle = contains.lower()
            pool = [i for i, r in enumerate(self.records) if needle in r.text.lower()]
            if not pool:
                raise ValueError(f"{self.name}: no record contains {contains!r}")
        else:
            pool = list(range(len(self)))

        if n is not None and n < len(pool):
            rng = random.Random(self.seed if seed is None else seed)
            pool = sorted(rng.sample(pool, n))

        return Selection(
            dataset=self.name,
            indices=pool,
            records=[self.records[i] for i in pool],
        )

    def sample(self, n: int, *, seed: int | None = None) -> Selection:
        return self.select(n=n, seed=seed)

    def batches(self, batch_size: int = 256) -> Iterator[list[Record]]:
        buf: list[Record] = []
        for rec in self.records:
            buf.append(rec)
            if len(buf) == batch_size:
                yield buf
                buf = []
        if buf:
            yield buf

    def to_jsonl(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for rec in self.records:
                fh.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
        return path

    def stats(self) -> dict[str, Any]:
        lengths = [len(r.text) for r in self.records]
        n = len(lengths) or 1
        return {
            "dataset": self.name,
            "size": len(lengths),
            "mean_chars": sum(lengths) / n,
            "min_chars": min(lengths, default=0),
            "max_chars": max(lengths, default=0),
        }
