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
from scipy.optimize import brentq
from scipy.stats import norm

try:  
    from models import EmbeddingSet
except Exception:  
    EmbeddingSet = None 


@dataclass
class CmagConfig:

    group_size: int = 100                  
    min_group_size: int | None = None     
    merge_overlapping: bool = True         

    gamma: float = 1e-7                  
    variant: str = "mahalanobis"           
    u_power: str = "sqrt"                  

    epsilon: float = 16.0                  
    delta_mode: str = "fixed"              
    delta: float = 1e-5                    
    delta_exponent: float = 15.0           
    assign: str = "centroid"               
    normalize_input: bool = True
    renormalize_output: bool = False

    victim_model: str | None = None        
    dataset: str | None = None            
    strict_dim_check: bool = True
    seed: int = 0
    device: str | None = None

    def __post_init__(self) -> None:
        if self.variant not in ("mahalanobis", "euclidean"):
            raise ValueError(f"variant must be mahalanobis|euclidean, got {self.variant!r}")
        if self.u_power not in ("sqrt", "inv_sqrt"):
            raise ValueError(f"u_power must be sqrt|inv_sqrt, got {self.u_power!r}")
        if self.delta_mode not in ("fixed", "power"):
            raise ValueError(f"delta_mode must be fixed|power, got {self.delta_mode!r}")
        if self.assign not in ("centroid", "nearest_member"):
            raise ValueError(f"assign must be centroid|nearest_member, got {self.assign!r}")
        if not 0.0 < self.delta < 1.0:
            raise ValueError(f"delta must lie in (0,1), got {self.delta}")
        if self.group_size < 2:
            raise ValueError(f"group_size must be >= 2, got {self.group_size}")

    def resolved_min_group_size(self) -> int:
        return max(2, self.group_size // 2 if self.min_group_size is None else self.min_group_size)

    def resolved_device(self) -> str:
        if self.device:
            return self.device
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"


def positive_definite_covariance(
    X: np.ndarray, gamma: float = 1e-7
) -> tuple[np.ndarray, int]:
    X = np.asarray(X, dtype=np.float64)
    m, n = X.shape
    if m < 2:
        raise ValueError(f"need at least 2 embeddings to form a covariance, got {m}")
    v = np.cov(X.T)
    w, p = np.linalg.eigh(v)             
    rank = int(np.linalg.matrix_rank(np.diag(w)))
    floor = np.concatenate([np.full(n - rank, gamma), np.zeros(rank)])
    v = (p * (w + floor)) @ p.T
    trace = float(np.trace(v))
    if trace <= 0:
        raise ValueError("degenerate covariance: non-positive trace")
    return (n / trace) * v, rank


def matrix_sqrt(sigma: np.ndarray, inverse: bool = False) -> np.ndarray:
    w, b = np.linalg.eigh(np.asarray(sigma, dtype=np.float64))
    w = np.maximum(w, 1e-300)
    root = 1.0 / np.sqrt(w) if inverse else np.sqrt(w)
    return (b * root) @ b.T


def ag_condition(d: float, epsilon: float, sigma: float) -> float:
    if sigma <= 0:
        return 1.0
    a = d / (2.0 * sigma) - epsilon * sigma
    b = -d / (2.0 * sigma) - epsilon * sigma
    return float(norm.cdf(a) - np.exp(epsilon * d + norm.logcdf(b)))


def analytic_gaussian_sigma(
    d: float, epsilon: float, delta: float, *, sigma_max: float = 1e12
) -> float:
    if d <= 0:
        return 0.0
    hi = 1.0
    while ag_condition(d, epsilon, hi) > delta:
        hi *= 2.0
        if hi > sigma_max:
            raise RuntimeError(
                f"no sigma below {sigma_max:g} satisfies f(d={d:g}, sigma) <= delta={delta:g} "
                f"at epsilon={epsilon:g}. delta is too small or epsilon too large."
            )
    return float(brentq(lambda s: ag_condition(d, epsilon, s) - delta, 1e-12, hi,
                        xtol=1e-13, rtol=8.9e-16))


def _topk_neighbours(X: torch.Tensor, k: int, chunk: int = 1024) -> np.ndarray:
    n = X.shape[0]
    k = min(k, n)
    out = np.empty((n, k), dtype=np.int64)
    for i in range(0, n, chunk):
        block = X[i : i + chunk]
        d = torch.cdist(block, X)
        out[i : i + block.shape[0]] = d.topk(k, dim=-1, largest=False).indices.cpu().numpy()
    return out


def build_covering(
    X: torch.Tensor,
    group_size: int = 100,
    *,
    min_group_size: int | None = None,
    verbose: bool = False,
) -> list[list[int]]:
    n = X.shape[0]
    if n < 2:
        raise ValueError(f"need at least 2 embeddings to build a covering, got {n}")
    group_size = min(group_size, n)
    floor = max(2, group_size // 2 if min_group_size is None else min_group_size)

    knn = _topk_neighbours(X, group_size)
    covered = np.zeros(n, dtype=bool)
    groups: list[list[int]] = []

    for size in range(group_size, 0, -1):
        while True:
            claimed_any = False
            for s in range(n):
                if covered[s]:
                    continue
                cand = knn[s][~covered[knn[s]]]
                if len(cand) == size:
                    groups.append(cand.tolist())
                    covered[cand] = True
                    claimed_any = True
            if not claimed_any:
                break
    if not covered.all():
        groups.append(np.flatnonzero(~covered).tolist())

    groups = _merge_small_groups(X, groups, floor, verbose=verbose)
    if verbose:
        sizes = [len(g) for g in groups]
        print(f"cmag covering: {len(groups)} group(s), sizes "
              f"min={min(sizes)} mean={np.mean(sizes):.1f} max={max(sizes)}")
    return groups


def _merge_small_groups(
    X: torch.Tensor, groups: list[list[int]], floor: int, *, verbose: bool = False
) -> list[list[int]]:
    keep = [g for g in groups if len(g) >= floor]
    small = [g for g in groups if len(g) < floor]
    if not small:
        return groups
    if not keep:
        merged = sorted(i for g in groups for i in g)
        if verbose:
            print(f"cmag every group below min_group_size={floor}; merged into one")
        return [merged]

    centroids = torch.stack([X[torch.as_tensor(g, device=X.device)].mean(0) for g in keep])
    for g in small:
        c = X[torch.as_tensor(g, device=X.device)].mean(0, keepdim=True)
        keep[int(torch.cdist(c, centroids).argmin())].extend(g)
    if verbose:
        print(f"cmag merged {len(small)} group(s) below min_group_size={floor}")
    return [sorted(g) for g in keep]


def merge_overlapping(groups: Sequence[Sequence[int]]) -> list[list[int]]:
    parent = list(range(len(groups)))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    seen: dict[int, int] = {}
    for j, g in enumerate(groups):
        for member in g:
            if member in seen:
                ra, rb = find(seen[member]), find(j)
                if ra != rb:
                    parent[rb] = ra
            else:
                seen[member] = j

    components: dict[int, set[int]] = {}
    for j, g in enumerate(groups):
        components.setdefault(find(j), set()).update(g)
    return [sorted(c) for c in components.values()]


def _vectors_of(x: Any) -> np.ndarray:
    if EmbeddingSet is not None and isinstance(x, EmbeddingSet):
        return x.vectors
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _to_tensor(x: Any, device: str | torch.device) -> torch.Tensor:
    return torch.as_tensor(_vectors_of(x), dtype=torch.float32, device=device)


class Cmag(nn.Module):
    def __init__(self, dim: int, config: CmagConfig | None = None, **overrides: Any) -> None:
        super().__init__()
        cfg = config or CmagConfig()
        for key, value in overrides.items():
            if not hasattr(cfg, key):
                raise TypeError(f"unknown CmagConfig field {key!r}")
            setattr(cfg, key, value)
        cfg.__post_init__()

        self.cfg = cfg
        self.dim = int(dim)
        self.device = torch.device(cfg.resolved_device())
        self._fitted = False

        self.groups: list[list[int]] = []
        self.centroids: torch.Tensor | None = None   # (n_groups, dim)
        self.U: torch.Tensor | None = None           # (n_groups, dim, dim)
        self.sigma: torch.Tensor | None = None       # (n_groups,)
        self.d0: torch.Tensor | None = None          # (n_groups,)
        self.ranks: list[int] = []
        self.span_energy: list[float] = []
        self.fit_vectors: torch.Tensor | None = None 
        self.fit_labels: torch.Tensor | None = None
        self.to(self.device)

    def fit(
        self,
        embeddings: Any,
        *,
        covering: Sequence[Sequence[int]] | None = None,
        verbose: bool = True,
    ) -> "Cmag":
        cfg = self.cfg
        e = _to_tensor(embeddings, self.device)
        if e.shape[1] != self.dim:
            raise ValueError(f"embedding dim {e.shape[1]} != this Cmag's dim {self.dim}")
        if cfg.normalize_input:
            e = F.normalize(e, dim=-1)

        groups = (
            build_covering(e, cfg.group_size, min_group_size=cfg.resolved_min_group_size(),
                           verbose=verbose)
            if covering is None
            else [list(int(i) for i in g) for g in covering]
        )
        if cfg.merge_overlapping:
            before = len(groups)
            groups = merge_overlapping(groups)
            if verbose and len(groups) != before:
                print(f"cmag Algorithm 3 merged {before} -> {len(groups)} component(s)")
        self.groups = groups

        x = e.double().cpu().numpy()
        u_list, sigma_list, d0_list, cent_list = [], [], [], []
        self.ranks, self.span_energy = [], []

        for j, g in enumerate(groups):
            block = x[np.asarray(g, dtype=int)]
            cent_list.append(block.mean(0))

            if cfg.variant == "euclidean":
                u = np.eye(self.dim)
                dist_basis = np.eye(self.dim)
                rank, span = self.dim, 1.0
            else:
                sigma_mat, rank = positive_definite_covariance(block, cfg.gamma)
                eig = np.linalg.eigvalsh(sigma_mat)
                span = float(eig[-rank:].sum() / eig.sum()) if rank else 0.0
                u = matrix_sqrt(sigma_mat, inverse=(cfg.u_power == "inv_sqrt"))
                dist_basis = matrix_sqrt(sigma_mat, inverse=True)
            u_list.append(u)
            self.ranks.append(rank)
            self.span_energy.append(span)

            proj = torch.as_tensor(block @ dist_basis)
            d0 = float(torch.cdist(proj, proj).max()) if len(g) > 1 else 0.0
            delta_j = (
                cfg.delta if cfg.delta_mode == "fixed"
                else float(len(g)) ** (-cfg.delta_exponent)
            )
            d0_list.append(d0)
            sigma_list.append(analytic_gaussian_sigma(d0, cfg.epsilon, delta_j))

            if verbose and (j % max(1, len(groups) // 10) == 0 or j == len(groups) - 1):
                print(f"cmag group {j + 1:4d}/{len(groups)}  m={len(g):4d}  rank={rank:4d}  "
                      f"d0={d0:.4f}  sigma={sigma_list[-1]:.4f}  span_energy={span:.6f}")

        self.centroids = torch.as_tensor(np.vstack(cent_list), dtype=torch.float32,
                                         device=self.device)
        self.U = torch.as_tensor(np.stack(u_list) if u_list else np.empty((0, self.dim, self.dim)),
                                 dtype=torch.float32, device=self.device)
        self.sigma = torch.as_tensor(sigma_list, dtype=torch.float32, device=self.device)
        self.d0 = torch.as_tensor(d0_list, dtype=torch.float32, device=self.device)

        if cfg.assign == "nearest_member":
            labels = np.empty(e.shape[0], dtype=np.int64)
            for j, g in enumerate(groups):
                labels[np.asarray(g, dtype=int)] = j
            self.fit_vectors = e.clone()
            self.fit_labels = torch.as_tensor(labels, device=self.device)

        self._fitted = True

        report = self.privacy_report()
        if report["snr_vs_unit_norm"] < 0.5:
            warnings.warn(
                f"CMAG: mean noise norm is {1 / report['snr_vs_unit_norm']:.0f}x the norm of a "
                f"unit-length embedding at epsilon={cfg.epsilon:g}, delta={cfg.delta:g}. Raise "
                "epsilon or delta. Note the noise is concentrated in the neighbourhood span "
                f"({report['noise_energy_in_span']:.4f} of its energy), so this is not uniform ",
                RuntimeWarning,
                stacklevel=2,
            )
        return self

    @torch.no_grad()
    def assign(self, e: torch.Tensor) -> torch.Tensor:
        if self.cfg.assign == "nearest_member" and self.fit_vectors is not None:
            return self.fit_labels[torch.cdist(e, self.fit_vectors).argmin(dim=-1)]
        return torch.cdist(e, self.centroids).argmin(dim=-1)

    @torch.no_grad()
    def protect(
        self,
        embeddings: Any,
        *,
        epsilon: float | None = None,
        seed: int | None = None,
    ) -> Any:
        if not self._fitted:
            raise RuntimeError("Cmag.protect() called before fit(); there is no covering yet.")
        e = _to_tensor(embeddings, self.device)
        if self.cfg.strict_dim_check and e.shape[1] != self.dim:
            raise ValueError(
                f"covering was fit for dim {self.dim} but got embeddings of dim {e.shape[1]}. "
                "A covering indexes one specific encoder's geometry and does not transfer."
            )
        if self.cfg.normalize_input:
            e = F.normalize(e, dim=-1)

        sigma = self.sigma if epsilon is None else self._sigma_at(epsilon)
        labels = self.assign(e)
        gen = None
        if seed is not None:
            gen = torch.Generator(device=self.device).manual_seed(int(seed))

        out = e.clone()
        for j in labels.unique().tolist():
            rows = labels == j
            z = torch.randn(int(rows.sum()), self.dim, device=self.device, generator=gen)
            out[rows] = e[rows] + float(sigma[j]) * (z @ self.U[j])
        if self.cfg.renormalize_output:
            out = F.normalize(out, dim=-1)

        meta_defense = {
            **asdict(self.cfg),
            "epsilon_requested": self.cfg.epsilon if epsilon is None else epsilon,
            "n_groups": len(self.groups),
            "noise_seed": seed,
        }
        if EmbeddingSet is not None and isinstance(embeddings, EmbeddingSet):
            return EmbeddingSet(
                vectors=out.cpu().numpy().astype(embeddings.vectors.dtype, copy=False),
                ids=list(embeddings.ids),
                indices=list(embeddings.indices),
                model=embeddings.model,
                dataset=embeddings.dataset,
                meta={**embeddings.meta, "defense": "cmag", "cmag": meta_defense},
            )
        if torch.is_tensor(embeddings):
            return out.to(embeddings.device, embeddings.dtype)
        return out.cpu().numpy()

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        return self.protect(e)

    def _sigma_at(self, epsilon: float) -> torch.Tensor:
        cfg = self.cfg
        vals = [
            analytic_gaussian_sigma(
                float(d),
                epsilon,
                cfg.delta if cfg.delta_mode == "fixed"
                else float(len(g)) ** (-cfg.delta_exponent),
            )
            for d, g in zip(self.d0.tolist(), self.groups)
        ]
        return torch.as_tensor(vals, dtype=torch.float32, device=self.device)


    @torch.no_grad()
    def privacy_report(self, epsilon: float | None = None) -> dict[str, Any]:
        if not self._fitted:
            raise RuntimeError("privacy_report() called before fit()")
        sigma = (self.sigma if epsilon is None else self._sigma_at(epsilon)).cpu()
        d0 = self.d0.cpu()
        sizes = [len(g) for g in self.groups]
        noise_norm = float((sigma * math.sqrt(self.dim)).mean())
        return {
            "epsilon": self.cfg.epsilon if epsilon is None else epsilon,
            "delta_mode": self.cfg.delta_mode,
            "delta": self.cfg.delta,
            "variant": self.cfg.variant,
            "dim": self.dim,
            "n_groups": len(self.groups),
            "group_size_min": int(min(sizes)),
            "group_size_mean": float(np.mean(sizes)),
            "group_size_max": int(max(sizes)),
            "sigma_rank_mean": float(np.mean(self.ranks)),
            "noise_energy_in_span": float(np.mean(self.span_energy)),
            "untouched_fraction": 1.0 - float(np.mean(self.ranks)) / self.dim,
            "d0_mean": float(d0.mean()),
            "d0_max": float(d0.max()),
            "sigma_min": float(sigma.min()),
            "sigma_mean": float(sigma.mean()),
            "sigma_max": float(sigma.max()),
            "expected_noise_norm": noise_norm,
            "snr_vs_unit_norm": 1.0 / noise_norm if noise_norm > 0 else float("inf"),
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
                "groups": self.groups,
                "centroids": self.centroids.cpu() if self.centroids is not None else None,
                "U": self.U.cpu() if self.U is not None else None,
                "sigma": self.sigma.cpu() if self.sigma is not None else None,
                "d0": self.d0.cpu() if self.d0 is not None else None,
                "ranks": self.ranks,
                "span_energy": self.span_energy,
                "fit_vectors": (
                    self.fit_vectors.cpu() if self.fit_vectors is not None else None
                ),
                "fit_labels": self.fit_labels.cpu() if self.fit_labels is not None else None,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: str | Path, *, device: str | None = None) -> "Cmag":
        ckpt = torch.load(Path(path), map_location="cpu", weights_only=False)
        cfg = CmagConfig(**ckpt["config"])
        if device:
            cfg.device = device
        obj = cls(ckpt["dim"], cfg)
        obj.groups = ckpt["groups"]
        obj.centroids = ckpt["centroids"].to(obj.device)
        obj.U = ckpt["U"].to(obj.device)
        obj.sigma = ckpt["sigma"].to(obj.device)
        obj.d0 = ckpt["d0"].to(obj.device)
        obj.ranks = ckpt["ranks"]
        obj.span_energy = ckpt["span_energy"]
        if ckpt.get("fit_vectors") is not None:
            obj.fit_vectors = ckpt["fit_vectors"].to(obj.device)
            obj.fit_labels = ckpt["fit_labels"].to(obj.device)
        obj._fitted = True
        return obj

    def __repr__(self) -> str:
        groups = len(self.groups) if self._fitted else -1
        sigma = f"{float(self.sigma.mean()):.4f}" if self._fitted else "n/a"
        return (
            f"Cmag(dim={self.dim}, variant={self.cfg.variant!r}, eps={self.cfg.epsilon}, "
            f"delta={self.cfg.delta:g}, groups={groups}, group_size={self.cfg.group_size}, "
            f"sigma_mean={sigma}, fitted={self._fitted})"
        )


CMAG = Cmag

if __name__ == "__main__":  
    print("=" * 78)

    print("\n" + "=" * 78)
    print("Correctness checks")
    print("=" * 78)

    def _reference_B(d, e, delta):
        def bf(s):
            return norm.cdf(d / (2 * s) - e * s) - math.exp(e * d) * norm.cdf(-d / (2 * s) - e * s)
        s = 1
        while bf(s) >= delta:
            s += 1
        s = s - 1
        for p in range(1, 11):
            i = 1
            while bf(s + pow(10, -p) * i) >= delta:
                i += 1
            if p < 10:
                i = i - 1
            s = s + pow(10, -p) * i
        return s

    worst = 0.0
    for d, e, dl in [(15.0, 1.6, 1e-5), (20.0, 8.0, 1e-8), (26.0, 16.0, 1e-12),
                     (5.0, 40.0, 1e-6), (0.3, 16.0, 1e-5)]:
        ref, got = _reference_B(d, e, dl), analytic_gaussian_sigma(d, e, dl)
        worst = max(worst, abs(ref - got) / ref)
        print(f"  sigma(d={d:5}, eps={e:5}, delta={dl:.0e}) = {got:.9f}  "
              f"(reference {ref:.9f})")
    print(f"  max relative deviation from the reference solver: {worst:.2e}")
    assert worst < 1e-8

    rng = np.random.default_rng(0)
    block = rng.normal(size=(32, 64))
    sig, rank = positive_definite_covariance(block)
    ev = np.linalg.eigvalsh(sig)
    print(f"\n  Algorithm 1: rank(V)={rank} (expect 31), tr(Sigma)={np.trace(sig):.6f} "
          f"(expect 64), min eig={ev[0]:.3e} > 0: {ev[0] > 0}")
    assert rank == 31 and abs(np.trace(sig) - 64) < 1e-8 and ev[0] > 0

    u = matrix_sqrt(sig)
    print(f"  ||U @ U - Sigma||_max = {np.abs(u @ u - sig).max():.3e}")
    assert np.abs(u @ u - sig).max() < 1e-8

    disjoint = [[0, 1, 2], [3, 4], [5]]
    assert merge_overlapping(disjoint) == [[0, 1, 2], [3, 4], [5]]
    assert merge_overlapping([[0, 1], [1, 2], [5, 6]]) == [[0, 1, 2], [5, 6]]
    print("  Algorithm 3: no-op on a disjoint covering, merges an overlapping one: True")

    print("\n" + "=" * 78)
    print("Behaviour on synthetic clustered embeddings")
    print("=" * 78)
    torch.manual_seed(0)
    n, d, dim_latent = 400, 128, 8
    latent = rng.normal(size=(n, dim_latent))
    base = latent @ rng.normal(size=(dim_latent, d)) + 0.3 * rng.normal(size=(n, d))
    base = (base / np.linalg.norm(base, axis=1, keepdims=True)).astype("float32")
    e0 = torch.as_tensor(base)

    guard = Cmag(d, group_size=50, epsilon=16.0, delta=1e-5, device="cpu", seed=0)
    guard.fit(e0, verbose=False)
    print(f"\n{guard!r}")
    print(f"privacy_report: {json.dumps(guard.privacy_report(), indent=2)}")

    e1 = guard.protect(e0, seed=0)
    print(f"utility_report: {json.dumps(guard.utility_report(e0, e1, k=10), indent=2)}")

    print(f"\n{'eps':>6} {'sigma_mean':>11} {'SNR':>8} {'recall@10':>10} {'spearman':>9}")
    for eps in (1.6, 8.0, 16.0, 40.0):
        rep = guard.privacy_report(epsilon=eps)
        util = guard.utility_report(e0, guard.protect(e0, epsilon=eps, seed=0), k=10)
        print(f"{eps:>6} {rep['sigma_mean']:>11.4f} {rep['snr_vs_unit_norm']:>8.4f} "
              f"{util['recall_at_10']:>10.3f} {util['sim_spearman']:>9.3f}")

    assert not torch.equal(guard.protect(e0), guard.protect(e0))
    assert torch.equal(guard.protect(e0, seed=7), guard.protect(e0, seed=7))
    print("\n  fresh noise per call, reproducible under a seed: True")

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        p = guard.save(Path(tmp) / "cmag.pt")
        g2 = Cmag.load(p, device="cpu")
        assert torch.equal(guard.protect(e0, seed=3), g2.protect(e0, seed=3))
        print("  save/load round-trip reproduces the mechanism exactly: True")

    ge = Cmag(d, group_size=50, epsilon=16.0, delta=1e-5, variant="euclidean",
              device="cpu", seed=0).fit(e0, verbose=False)
    print(f"\n  CMAG(E): {ge!r}")
    print(f"           noise_energy_in_span = {ge.privacy_report()['noise_energy_in_span']} "
          f"(1.0 by construction: U = I)")
