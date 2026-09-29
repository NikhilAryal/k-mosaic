from __future__ import annotations

import json
import math
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:  
    from models import EmbeddingSet
except Exception: 
    EmbeddingSet = None


@dataclass
class IdctConfig:
    num_subsets: int = 2               
    wrap_last: bool = False           
    dct_norm: str = "ortho"            
    resample: str = "never"         

    normalize_input: bool = True
    renormalize_output: bool = False

    victim_model: str | None = None   
    dataset: str | None = None        
    strict_dim_check: bool = True     
    seed: int = 0
    device: str | None = None

    def __post_init__(self) -> None:
        if self.dct_norm not in ("ortho", "backward"):
            raise ValueError(f"dct_norm must be 'ortho' or 'backward', got {self.dct_norm!r}")
        if self.resample not in ("never", "per_call"):
            raise ValueError(f"resample must be 'never' or 'per_call', got {self.resample!r}")
        if self.num_subsets < 1:
            raise ValueError(f"num_subsets (K) must be >= 1, got {self.num_subsets}")

    def resolved_device(self) -> str:
        if self.device:
            return self.device
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"


def dct_matrix(dim: int, norm: str = "ortho", dtype: torch.dtype = torch.float64) -> torch.Tensor:
    n = torch.arange(dim, dtype=dtype)
    k = n.unsqueeze(1)
    w = torch.cos(math.pi * (2.0 * n + 1.0) * k / (2.0 * dim))
    if norm == "ortho":
        scale = torch.full((dim, 1), math.sqrt(2.0 / dim), dtype=dtype)
        scale[0] = math.sqrt(1.0 / dim)
        return w * scale
    return 2.0 * w


def partition_indices(dim: int, num_subsets: int, seed: int = 0) -> list[list[int]]:
    if num_subsets > dim:
        raise ValueError(f"num_subsets={num_subsets} exceeds dim={dim}")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(dim)
    return [chunk.tolist() for chunk in np.array_split(perm, num_subsets)]


def overlap_matrix(
    dim: int,
    subset: Sequence[int],
    *,
    wrap_last: bool = False,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    v = list(int(i) for i in subset)
    length = len(v)
    if length < 2 and not (wrap_last and length == 1):
        raise ValueError(
            f"subset of size {length} yields an all-zero overlap matrix (the literal "
            "'excluding the last index' rule leaves nothing), which would release "
            "identically-zero embeddings. Lower num_subsets. "
            "See AMBIGUITIES['degenerate_subsets']."
        )
    m = torch.zeros((dim, dim), dtype=dtype)
    upto = length if wrap_last else length - 1
    for j in range(upto):
        a, b = v[j], v[(j + 1) % length]
        m[a, a] = 1.0
        m[a, b] = 1.0
    return m.T.contiguous()


def transform_matrix(
    dim: int,
    overlap: torch.Tensor,
    *,
    dct_norm: str = "ortho",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    w = dct_matrix(dim, dct_norm, dtype=dtype)
    o = overlap.to(dtype)
    if dct_norm == "ortho":
        w_inv_t = w                      
    else:
        w_inv_t = torch.linalg.inv(w).T  
    return w.T @ o @ w_inv_t


def _vectors_of(x: Any) -> np.ndarray:
    if EmbeddingSet is not None and isinstance(x, EmbeddingSet):
        return x.vectors
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _to_tensor(x: Any, device: str | torch.device) -> torch.Tensor:
    return torch.as_tensor(_vectors_of(x), dtype=torch.float32, device=device)


class IdctDefense(nn.Module):
    def __init__(
        self,
        dim: int,
        config: IdctConfig | None = None,
        *,
        subsets: Sequence[Sequence[int]] | None = None,
        selected: int | None = None,
        **overrides: Any,
    ) -> None:
        super().__init__()
        cfg = config or IdctConfig()
        for key, value in overrides.items():
            if not hasattr(cfg, key):
                raise TypeError(f"unknown IdctConfig field {key!r}")
            setattr(cfg, key, value)
        cfg.__post_init__()

        self.cfg = cfg
        self.dim = int(dim)
        self.device = torch.device(cfg.resolved_device())

        self.subsets: list[list[int]] = (
            [list(int(i) for i in s) for s in subsets]
            if subsets is not None
            else partition_indices(self.dim, cfg.num_subsets, cfg.seed)
        )
        if sorted(i for s in self.subsets for i in s) != list(range(self.dim)):
            raise ValueError("subsets must be a partition of {0, ..., dim-1}")

        self._overlaps = [
            overlap_matrix(self.dim, s, wrap_last=cfg.wrap_last) for s in self.subsets
        ]

        self._gen = np.random.default_rng(cfg.seed + 1)
        self.selected = int(selected) if selected is not None else int(
            self._gen.integers(len(self._overlaps))
        )

        self.register_buffer("A", self._build_A(self.selected), persistent=False)
        self.to(self.device)

    def _build_A(self, index: int) -> torch.Tensor:
        a = transform_matrix(
            self.dim, self._overlaps[index], dct_norm=self.cfg.dct_norm
        )
        return a.to(torch.float32)

    @classmethod
    def from_subsets(
        cls,
        dim: int,
        subsets: Sequence[Sequence[int]],
        *,
        selected: int = 0,
        **overrides: Any,
    ) -> "IdctDefense":
        cfg = IdctConfig(num_subsets=len(list(subsets)))
        return cls(dim, cfg, subsets=subsets, selected=selected, **overrides)

    @property
    def overlap(self) -> torch.Tensor:
        return self._overlaps[self.selected].to(torch.float32)

    @property
    def transform(self) -> torch.Tensor:
        return self.A

    @torch.no_grad()
    def protect(self, embeddings: Any, *, batch_size: int = 8192) -> Any:
        e = _to_tensor(embeddings, self.device)
        if self.cfg.strict_dim_check and e.shape[1] != self.dim:
            raise ValueError(
                f"O_s was built for dim {self.dim} but got embeddings of dim {e.shape[1]}. "
            )
        if self.cfg.normalize_input:
            e = F.normalize(e, dim=-1)

        if self.cfg.resample == "per_call":
            self.selected = int(self._gen.integers(len(self._overlaps)))
            self.A = self._build_A(self.selected).to(self.device)

        a = self.A.to(self.device, e.dtype)
        out = torch.cat([e[i : i + batch_size] @ a for i in range(0, e.shape[0], batch_size)])
        if self.cfg.renormalize_output:
            out = F.normalize(out, dim=-1)

        meta_defense = {**asdict(self.cfg), "selected_subset": self.selected,
                        "subset_size": len(self.subsets[self.selected]),
                        "rank": self.rank()}
        if EmbeddingSet is not None and isinstance(embeddings, EmbeddingSet):
            return EmbeddingSet(
                vectors=out.cpu().numpy().astype(embeddings.vectors.dtype, copy=False),
                ids=list(embeddings.ids),
                indices=list(embeddings.indices),
                model=embeddings.model,
                dataset=embeddings.dataset,
                meta={**embeddings.meta, "defense": "idct", "idct": meta_defense},
            )
        if torch.is_tensor(embeddings):
            return out.to(embeddings.device, embeddings.dtype)
        return out.cpu().numpy()

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        return self.protect(e)

    @torch.no_grad()
    def rank(self) -> int:
        return int(torch.linalg.matrix_rank(self._overlaps[self.selected]))

    @torch.no_grad()
    def rank_report(self) -> dict[str, Any]:
        a = self.A.double().cpu()
        svals = torch.linalg.svdvals(a)
        rank = self.rank()
        sv_min_nonzero = float(svals[rank - 1]) if rank else 0.0
        sizes = [len(s) for s in self.subsets]
        return {
            "dim": self.dim,
            "num_subsets": self.cfg.num_subsets,
            "subset_sizes": sizes,
            "selected_subset": self.selected,
            "selected_size": sizes[self.selected],
            "wrap_last": self.cfg.wrap_last,
            "dct_norm": self.cfg.dct_norm,
            "rank": rank,
            "rank_fraction": rank / self.dim,
            "null_fraction": 1.0 - rank / self.dim,
            "expected_energy_retained": float((a**2).sum() / self.dim),
            "sv_max": float(svals[0]),
            "sv_min_nonzero": sv_min_nonzero,
        }

    @torch.no_grad()
    def utility_report(self, original: Any, protected: Any, *, k: int = 10) -> dict[str, float]:
        a = F.normalize(_to_tensor(original, self.device), dim=-1)
        b = F.normalize(_to_tensor(protected, self.device), dim=-1)
        n = a.shape[0]
        if b.shape[0] != n:
            raise ValueError(f"row count mismatch: {n} vs {b.shape[0]}")

        sim_a, sim_b = a @ a.t(), b @ b.t()
        eye = torch.eye(n, device=self.device, dtype=torch.bool)
        neg_inf = torch.finfo(sim_a.dtype).min
        kk = min(k, n - 1)
        top_a = sim_a.masked_fill(eye, neg_inf).topk(kk, dim=-1).indices
        top_b = sim_b.masked_fill(eye, neg_inf).topk(kk, dim=-1).indices
        overlap = [len(set(top_a[i].tolist()) & set(top_b[i].tolist())) / kk for i in range(n)]

        off = ~eye
        flat_a = sim_a[off].cpu().numpy()
        flat_b = sim_b[off].cpu().numpy()
        rank_a = np.argsort(np.argsort(flat_a))
        rank_b = np.argsort(np.argsort(flat_b))
        return {
            f"recall_at_{kk}": float(np.mean(overlap)),
            "sim_pearson": float(np.corrcoef(flat_a, flat_b)[0, 1]),
            "sim_spearman": float(np.corrcoef(rank_a, rank_b)[0, 1]),
            "mean_abs_sim_error": float(np.mean(np.abs(flat_a - flat_b))),
            "n": float(n),
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "dim": self.dim,
                "config": asdict(self.cfg),
                "subsets": self.subsets,
                "selected": self.selected,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: str | Path, *, device: str | None = None) -> "IdctDefense":
        ckpt = torch.load(Path(path), map_location="cpu", weights_only=False)
        cfg = IdctConfig(**ckpt["config"])
        if device:
            cfg.device = device
        return cls(
            ckpt["dim"], cfg, subsets=ckpt["subsets"], selected=ckpt["selected"]
        )

    def __repr__(self) -> str:
        return (
            f"IdctDefense(dim={self.dim}, K={self.cfg.num_subsets}, "
            f"s={self.selected}, |v_s|={len(self.subsets[self.selected])}, "
            f"rank={self.rank()}/{self.dim}, dct_norm={self.cfg.dct_norm!r}, "
            f"wrap_last={self.cfg.wrap_last}, resample={self.cfg.resample!r})"
        )


IDCT = IdctDefense

if __name__ == "__main__":  

    # for name, item in AMBIGUITIES.items():
    #     print(f"\n[{name}]")
    #     for field_name in ("paper", "problem", "assumed", "knob"):
    #         print(f"  {field_name:8s}: {item[field_name]}")

    print("\n" + "=" * 78)
    print("Correctness checks")
    print("=" * 78)

    from scipy.fft import dct as _sdct, idct as _sidct

    rng = np.random.default_rng(0)
    d = 64

    x = rng.normal(size=d)
    for norm in ("ortho", "backward"):
        w = dct_matrix(d, norm).numpy()
        err = np.abs(w @ x - _sdct(x, type=2, norm=norm)).max()
        print(f"  dct_matrix vs scipy.fft.dct(norm={norm!r}): max err {err:.3e}")
        assert err < 1e-10, norm


    v = [5, 2, 9, 1]
    o = overlap_matrix(d, v)
    c = rng.normal(size=d)
    got = c @ o.numpy()
    want = np.zeros(d)
    for j in range(len(v) - 1):
        want[v[j]] = c[v[j]] + c[v[j + 1]]
    print(f"  O_s action == 'sum with next bin, drop the rest': "
          f"max err {np.abs(got - want).max():.3e}")
    assert np.abs(got - want).max() < 1e-12

    for norm in ("ortho", "backward"):
        guard = IdctDefense(d, num_subsets=4, seed=0, dct_norm=norm,
                            device="cpu", normalize_input=False)
        e = rng.normal(size=(32, d))
        literal = _sidct(_sdct(e, type=2, norm=norm, axis=-1) @ guard.overlap.numpy(),
                         type=2, norm=norm, axis=-1)
        fast = guard.protect(e.astype("float32"))
        err = np.abs(literal - fast).max()
        print(f"  A vs literal IDCT(DCT(E)·O_s) [norm={norm!r}]: max err {err:.3e}")
        assert err < 1e-4, norm

    print("\n" + "=" * 78)
    print("What the defense costs, as a function of the unspecified K")
    print("=" * 78)

    n = 256
    latent = rng.normal(size=(n, 16))
    base = latent @ rng.normal(size=(16, d)) + 0.35 * rng.normal(size=(n, d))
    base = (base / np.linalg.norm(base, axis=1, keepdims=True)).astype("float32")
    e0 = torch.as_tensor(base)

    print(f"{'K':>3} {'|v_s|':>6} {'rank':>5} {'null_frac':>10} "
          f"{'energy':>8} {'recall@10':>10} {'spearman':>9}")
    for k_subsets in (1, 2, 4, 8, 16):
        g = IdctDefense(d, num_subsets=k_subsets, seed=0, device="cpu")
        rep = g.rank_report()
        util = g.utility_report(e0, g.protect(e0), k=10)
        print(f"{k_subsets:>3} {rep['selected_size']:>6} {rep['rank']:>5} "
              f"{rep['null_fraction']:>10.3f} {rep['expected_energy_retained']:>8.3f} "
              f"{util['recall_at_10']:>10.3f} {util['sim_spearman']:>9.3f}")

    g = IdctDefense(d, num_subsets=2, seed=0, device="cpu")
    print(f"\n{g!r}")
    print(f"rank_report: {json.dumps(g.rank_report(), indent=2)}")

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        p = g.save(Path(tmp) / "idct.pt")
        g2 = IdctDefense.load(p, device="cpu")
        assert torch.equal(g.A, g2.A), "checkpoint round-trip changed A"
        print("\n  save/load round-trip reproduces A exactly: True")

    ident = IdctDefense(d, num_subsets=1, wrap_last=True, seed=0, device="cpu")
    print(f"  K=1, wrap_last=True -> rank {ident.rank()}/{d} "
          f"(cyclic bidiagonal is singular for even L)")
