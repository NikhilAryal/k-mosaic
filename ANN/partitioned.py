from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .index import IndexSpec, empty_index, prepare, set_budget

NPROBE_GRID = (1, 2, 4, 8, 16, 32, 64, 128, 256)


@dataclass(frozen=True)
class PartitionSpec:

    m: int = 1000
    alpha: float = 1.3
    branching: int = 8
    candidates: int = 16
    nprobe: int = 16
    key: str = "keyed-rotation-42"
    rotate: bool = True

    def config(self, metric: str, seed: int):
        from defense.k_mosaic import KMosaicConfig

        return KMosaicConfig(m=self.m, alpha=self.alpha, branching=self.branching,
                                   candidates=self.candidates, key=self.key, seed=seed,
                                   metric=metric)

    def storage_tag(self) -> str:
        from defense.k_mosaic import PRF_VERSION

        tag = f"kr{PRF_VERSION}_m{self.m}_a{self.alpha:g}_b{self.branching}"
        return tag if self.rotate else tag + "_norot"

    def utility_tag(self) -> str:
        return f"kr_m{self.m}_a{self.alpha:g}_b{self.branching}_np{self.nprobe}"

    def tag(self) -> str:
        return self.utility_tag()


def check_flat(spec: IndexSpec) -> None:
    if spec.factory.split(",")[-1] != "Flat":
        raise SystemExit(
            f"--partition needs flat per-cell storage (e.g. HNSW32,Flat or Flat), got "
            f"{spec.factory!r}: a trained codec per cell is not implemented."
        )


class PartitionedIndex:
    def __init__(self, kr: Any, spec: IndexSpec, pspec: PartitionSpec) -> None:
        self.kr, self.spec, self.pspec = kr, spec, pspec
        self.cells: dict[int, tuple[Any, np.ndarray]] = {}

    @property
    def budget_param(self) -> str:
        return "nprobe"

    @property
    def budget(self) -> int:
        return min(self.pspec.nprobe, self.kr.n_cells)

    def grid(self) -> list[int]:
        C = self.kr.n_cells
        return sorted({b for b in NPROBE_GRID if b <= C} | {self.budget})

    def report(self) -> dict[str, Any]:
        rep = self.kr.report()
        rep["search"] = f"nprobe={self.budget}, efSearch={self.spec.budget}"
        return rep

    @classmethod
    def build(cls, X: np.ndarray, spec: IndexSpec, pspec: PartitionSpec, *, seed: int = 42,
              device: str | None = None) -> "PartitionedIndex":
        from defense.k_mosaic import KMosaic

        check_flat(spec)
        kr = KMosaic(pspec.config(spec.metric, seed), device=device).fit(X)
        out = cls(kr, spec, pspec)
        stored = kr.protect(X, kr.labels) if pspec.rotate else X
        order = np.argsort(kr.labels, kind="stable")
        bounds = np.searchsorted(kr.labels[order], np.arange(kr.n_cells + 1))
        jobs = [(c, order[bounds[c]:bounds[c + 1]]) for c in range(kr.n_cells)
                if bounds[c + 1] > bounds[c]]

        def one(job):
            c, ids = job
            _single_thread()
            sub = empty_index(spec, X.shape[1])
            sub.add(np.ascontiguousarray(stored[ids]))
            set_budget(sub, spec)
            return c, sub, ids

        with ThreadPoolExecutor(max_workers=_workers()) as pool:
            for c, sub, ids in pool.map(one, jobs):
                out.cells[c] = (sub, ids)
        return out

    def search(self, Q: np.ndarray, k: int, budget: int | None = None) -> np.ndarray:
        nprobe = self.budget if budget is None else min(int(budget), self.kr.n_cells)
        Q = np.ascontiguousarray(Q, dtype=np.float32)
        route = self.kr.route(Q, nprobe=nprobe).reshape(len(Q), -1)
        scores = np.full((len(Q), nprobe * k), -np.inf, dtype=np.float32)
        ids = np.full((len(Q), nprobe * k), -1, dtype=np.int64)
        Qt = torch.as_tensor(Q, device=self.kr.device)

        tasks = []
        for c in np.unique(route):
            qi, slot = np.nonzero(route == c)
            if int(c) in self.cells:
                q = self.kr.rotate(Qt[qi], np.full(len(qi), c)) if self.pspec.rotate else Qt[qi]
                tasks.append((int(c), qi, slot, q.cpu().numpy()))

        def one(task):
            c, qi, slot, q = task
            _single_thread()
            sub, gid = self.cells[c]
            D, I = sub.search(q, k)
            return qi, slot, D, np.where(I >= 0, gid[np.maximum(I, 0)], -1)

        with ThreadPoolExecutor(max_workers=_workers()) as pool:
            for qi, slot, D, G in pool.map(one, tasks):
                cols = slot[:, None] * k + np.arange(k)[None, :]
                scores[qi[:, None], cols] = np.where(G >= 0, D, -np.inf)
                ids[qi[:, None], cols] = G
        top = np.argsort(-scores, axis=1, kind="stable")[:, :k]
        return np.take_along_axis(ids, top, axis=1)


class PartitionedStorage:
    lossless = True

    def __init__(self, kr: Any, spec: IndexSpec, pspec: PartitionSpec) -> None:
        self.kr, self.spec, self.pspec = kr, spec, pspec
        self.last_cells: np.ndarray | None = None

    def __call__(self, X: Any) -> Any:
        is_t = torch.is_tensor(X)
        arr = X.detach().float().cpu().numpy() if is_t else np.asarray(X, dtype=np.float32)
        P = prepare(arr, self.spec.metric)
        cells = self.kr.route(P)
        self.last_cells = cells
        out = self.kr.protect(P, cells) if self.pspec.rotate else P
        return torch.as_tensor(out, dtype=X.dtype, device=X.device) if is_t else out

    def describe(self) -> str:
        r = self.kr.report()
        rot = "k_mosaic per cell" if self.pspec.rotate else "NO rotation (control)"
        return (f"partitioned {self.spec.factory} [{self.spec.metric}] storage, "
                f"{r['cells_effective']} cells (m={self.pspec.m}, alpha={self.pspec.alpha:g}), "
                f"{rot}")

    def __repr__(self) -> str:
        return f"PartitionedStorage({self.describe()})"


def _single_thread() -> None:
    import faiss

    faiss.omp_set_num_threads(1)


def _workers() -> int:
    return max(1, min(128, os.cpu_count() or 1))


__all__ = ["NPROBE_GRID", "PartitionSpec", "PartitionedIndex", "PartitionedStorage",
           "check_flat"]
