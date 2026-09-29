from __future__ import annotations

import json
import math
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:  
    from models import EmbeddingSet
except Exception: 
    EmbeddingSet = None 


@dataclass
class RemoteRagConfig:

    budget_mode: str = "radius"            
    radius: float = 0.05                   
    epsilon: float = 10.0                  
    gamma_shape: int | None = None        

    normalize_input: bool = True
    renormalize_output: bool = False

    victim_model: str | None = None 
    dataset: str | None = None 
    strict_dim_check: bool = True
    seed: int = 0
    device: str | None = None

    def __post_init__(self) -> None:
        if self.budget_mode not in ("radius", "per_dim", "absolute"):
            raise ValueError(
                f"budget_mode must be radius|per_dim|absolute, got {self.budget_mode!r}"
            )
        if self.budget_mode == "radius" and self.radius <= 0:
            raise ValueError(f"radius (r_bar) must be > 0, got {self.radius}")
        if self.budget_mode != "radius" and self.epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got {self.epsilon}")

    def resolved_device(self) -> str:
        if self.device:
            return self.device
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"


def epsilon_for(dim: int, cfg: RemoteRagConfig, radius: float | None = None) -> float:
    r = cfg.radius if radius is None else float(radius)
    if cfg.budget_mode == "radius":
        if r <= 0:
            raise ValueError(f"radius must be > 0, got {r}")
        return dim / r
    if cfg.budget_mode == "per_dim":
        return cfg.epsilon * dim
    return cfg.epsilon


def mean_radius(dim: int, epsilon: float) -> float:
    return dim / epsilon


def distance_dp_noise(
    n_rows: int,
    dim: int,
    epsilon: float,
    *,
    shape: int | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if epsilon <= 0:
        raise ValueError(f"epsilon must be > 0, got {epsilon}")
    k = float(shape if shape is not None else dim)

    direction = torch.randn(n_rows, dim, device=device, dtype=dtype, generator=generator)
    direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    if generator is None:
        radius = torch.distributions.Gamma(
            torch.tensor(k, device=device, dtype=dtype),
            torch.tensor(float(epsilon), device=device, dtype=dtype),
        ).sample((n_rows,))
    else:
        seed = int(torch.randint(
            0, 2**31 - 1, (1,), generator=generator, device=generator.device
        ).item())
        draw = np.random.default_rng(seed).gamma(shape=k, scale=1.0 / epsilon, size=n_rows)
        radius = torch.as_tensor(draw, device=device, dtype=dtype)

    return radius.unsqueeze(-1) * direction

def _vectors_of(x: Any) -> np.ndarray:
    if EmbeddingSet is not None and isinstance(x, EmbeddingSet):
        return x.vectors
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _to_tensor(x: Any, device: str | torch.device) -> torch.Tensor:
    return torch.as_tensor(_vectors_of(x), dtype=torch.float32, device=device)


class RemoteRag(nn.Module):
    def __init__(self, dim: int, config: RemoteRagConfig | None = None, **overrides: Any) -> None:
        super().__init__()
        cfg = config or RemoteRagConfig()
        for key, value in overrides.items():
            if not hasattr(cfg, key):
                raise TypeError(f"unknown RemoteRagConfig field {key!r}")
            setattr(cfg, key, value)
        cfg.__post_init__()

        self.cfg = cfg
        self.dim = int(dim)
        self.device = torch.device(cfg.resolved_device())
        self._fitted = False
        self.corpus_norm_mean: float = 1.0
        self.corpus_norm_std: float = 0.0
        self.to(self.device)

    def fit(self, embeddings: Any = None, *, verbose: bool = True) -> "RemoteRag":
        if embeddings is not None:
            e = _to_tensor(embeddings, self.device)
            if e.shape[1] != self.dim:
                raise ValueError(f"embedding dim {e.shape[1]} != this defense's dim {self.dim}")
            norms = e.norm(dim=-1)
            self.corpus_norm_mean = float(norms.mean())
            self.corpus_norm_std = float(norms.std()) if norms.numel() > 1 else 0.0
            if verbose:
                print(f"remote_rag corpus ||e||: mean={self.corpus_norm_mean:.4f} "
                      f"std={self.corpus_norm_std:.4f}  (n={len(norms)})")
        self._fitted = True

        report = self.privacy_report()
        if verbose:
            print(
                f"remote_rag budget_mode={self.cfg.budget_mode!r} -> "
                f"eps={report['epsilon']:.1f}, r_bar={report['mean_radius']:.4f}, "
                f"angle={report['expected_angle_shift_deg']:.2f} deg"
            )
        if report["radius_to_signal"] > 1.0:
            warnings.warn(
                f"remote_rag: mean radius {report['mean_radius']:.3f} exceeds the signal norm "
                f"{report['signal_norm']:.3f}. ",
                RuntimeWarning,
                stacklevel=2,
            )
        return self

    @torch.no_grad()
    def protect(
        self,
        embeddings: Any,
        *,
        radius: float | None = None,
        epsilon: float | None = None,
        seed: int | None = None,
    ) -> Any:
        e = _to_tensor(embeddings, self.device)
        if self.cfg.strict_dim_check and e.shape[1] != self.dim:
            raise ValueError(
                f"defense was built for dim {self.dim} but got embeddings of dim {e.shape[1]}."
            )
        if self.cfg.normalize_input:
            e = F.normalize(e, dim=-1)

        eps = (
            float(epsilon) if epsilon is not None
            else epsilon_for(self.dim, self.cfg, radius)
        )
        gen = None
        if seed is not None:
            gen = torch.Generator(device=self.device).manual_seed(int(seed))

        out = e + distance_dp_noise(
            e.shape[0], self.dim, eps,
            shape=self.cfg.gamma_shape, device=self.device, dtype=e.dtype, generator=gen,
        )
        if self.cfg.renormalize_output:
            out = F.normalize(out, dim=-1)

        meta_defense = {
            **asdict(self.cfg),
            "epsilon_effective": eps,
            "mean_radius": mean_radius(self.dim, eps),
            "noise_seed": seed,
        }
        if EmbeddingSet is not None and isinstance(embeddings, EmbeddingSet):
            return EmbeddingSet(
                vectors=out.cpu().numpy().astype(embeddings.vectors.dtype, copy=False),
                ids=list(embeddings.ids),
                indices=list(embeddings.indices),
                model=embeddings.model,
                dataset=embeddings.dataset,
                meta={**embeddings.meta, "defense": "remote_rag", "remote_rag": meta_defense},
            )
        if torch.is_tensor(embeddings):
            return out.to(embeddings.device, embeddings.dtype)
        return out.cpu().numpy()

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        return self.protect(e)

    @torch.no_grad()
    def privacy_report(
        self, radius: float | None = None, epsilon: float | None = None
    ) -> dict[str, Any]:

        eps = (
            float(epsilon) if epsilon is not None
            else epsilon_for(self.dim, self.cfg, radius)
        )
        n = self.dim
        r_bar = mean_radius(n, eps)
        signal = self.corpus_norm_mean if not self.cfg.normalize_input else 1.0

        angle = math.atan2(r_bar, signal)
        return {
            "budget_mode": self.cfg.budget_mode,
            "epsilon": eps,
            "epsilon_per_dim": eps / n,
            "dim": n,
            "gamma_shape": self.cfg.gamma_shape or n,
            "mean_radius": r_bar,
            "radius_rel_std": 1.0 / math.sqrt(self.cfg.gamma_shape or n),
            "signal_norm": signal,
            "radius_to_signal": r_bar / signal if signal > 0 else float("inf"),
            "snr": signal / r_bar if r_bar > 0 else float("inf"),
            "expected_angle_shift_rad": angle,
            "expected_angle_shift_deg": math.degrees(angle),
            "corpus_norm_mean": self.corpus_norm_mean,
            "stateless": True,
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
                "corpus_norm_mean": self.corpus_norm_mean,
                "corpus_norm_std": self.corpus_norm_std,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: str | Path, *, device: str | None = None) -> "RemoteRag":
        ckpt = torch.load(Path(path), map_location="cpu", weights_only=False)
        cfg = RemoteRagConfig(**ckpt["config"])
        if device:
            cfg.device = device
        obj = cls(ckpt["dim"], cfg)
        obj.corpus_norm_mean = ckpt.get("corpus_norm_mean", 1.0)
        obj.corpus_norm_std = ckpt.get("corpus_norm_std", 0.0)
        obj._fitted = True
        return obj

    def __repr__(self) -> str:
        rep = self.privacy_report()
        return (
            f"RemoteRag(dim={self.dim}, mode={self.cfg.budget_mode!r}, "
            f"eps={rep['epsilon']:.1f}, r_bar={rep['mean_radius']:.4f}, "
            f"angle={rep['expected_angle_shift_deg']:.2f}deg)"
        )

DistanceDP = RemoteRag

if __name__ == "__main__": 
    print("\n" + "=" * 78)
    print("Correctness checks")
    print("=" * 78)
    torch.manual_seed(0)
    d = 768

    for eps in (7680.0, 25600.0):
        z = distance_dp_noise(20000, d, eps)
        r = z.norm(dim=-1)
        print(f"  eps={eps:>8.0f}  E[r] measured {float(r.mean()):.5f}  "
              f"predicted n/eps {d / eps:.5f}  rel-std {float(r.std() / r.mean()):.4f} "
              f"(1/sqrt(n) = {1 / math.sqrt(d):.4f})")
        assert abs(float(r.mean()) - d / eps) / (d / eps) < 0.02
        assert abs(float(r.std() / r.mean()) - 1 / math.sqrt(d)) < 0.005

    v = distance_dp_noise(20000, d, 7680.0)
    v = v / v.norm(dim=-1, keepdim=True)
    print(f"  direction uniformity: max|mean_i| {float(v.mean(0).abs().max()):.4f}, "
          f"E[v_i^2] {float((v ** 2).mean()):.3e} (1/n = {1 / d:.3e})")
    assert float(v.mean(0).abs().max()) < 0.05

    from attacker.algen.defenses import lap_mech

    e = F.normalize(torch.randn(20000, d), dim=-1)
    eps = 7680.0
    mine = (RemoteRag(d, budget_mode="absolute", epsilon=eps, device="cpu").protect(e) - e)
    theirs = lap_mech(e, eps) - e
    rm, rt = mine.norm(dim=-1), theirs.norm(dim=-1)
    print(f"\n  vs lap_mech(eps={eps:.0f}): E[r] {float(rm.mean()):.5f} vs "
          f"{float(rt.mean()):.5f}, std {float(rm.std()):.5f} vs {float(rt.std()):.5f}")
    assert abs(float(rm.mean()) - float(rt.mean())) / float(rt.mean()) < 0.02
    assert abs(float(rm.std()) - float(rt.std())) / float(rt.std()) < 0.05
    print("  => same mechanism; this file contributes the operating point, not the sampler")

    cfg_r = RemoteRagConfig(budget_mode="radius", radius=0.1)
    cfg_p = RemoteRagConfig(budget_mode="per_dim", epsilon=10.0)   # Figure 2's eps = 10n
    print(f"\n  radius=0.1 -> eps {epsilon_for(d, cfg_r):.1f}; "
          f"per_dim eps=10 -> eps {epsilon_for(d, cfg_p):.1f}  (Figure 2's eps = 10n)")
    assert abs(epsilon_for(d, cfg_r) - epsilon_for(d, cfg_p)) < 1e-6

    print("\n" + "=" * 78)
    print("Table 6's sweep on unit-norm embeddings (n=768, the paper's own gtr-t5-base)")
    print("=" * 78)
    rng = np.random.default_rng(0)
    latent = rng.normal(size=(400, 24))
    base = latent @ rng.normal(size=(24, d)) + 0.3 * rng.normal(size=(400, d))
    base = (base / np.linalg.norm(base, axis=1, keepdims=True)).astype("float32")
    e0 = torch.as_tensor(base)

    guard = RemoteRag(d, device="cpu").fit(e0, verbose=False)
    print(f"{'r':>8} {'eps = n/r':>10} {'angle deg':>10} {'recall@10':>10} {'spearman':>9}")
    for r in (0.03, 0.05, 0.07, 0.1):
        rep = guard.privacy_report(radius=r)
        util = guard.utility_report(e0, guard.protect(e0, radius=r, seed=0), k=10)
        print(f"{r:>8} {rep['epsilon']:>10.0f} {rep['expected_angle_shift_deg']:>10.2f} "
              f"{util['recall_at_10']:>10.3f} {util['sim_spearman']:>9.3f}")

    print(f"\n  the identical sampler at SPARSE's published eps:")
    for eps in (5.0, 40.0):
        rep = guard.privacy_report(epsilon=eps)
        print(f"    eps={eps:<5g} -> r_bar {rep['mean_radius']:>7.2f}  "
              f"({rep['radius_to_signal']:.0f}x a unit-norm embedding)")

    print(f"\n{guard!r}")
    print(f"privacy_report: {json.dumps(guard.privacy_report(), indent=2)}")

    assert not torch.equal(guard.protect(e0), guard.protect(e0))
    assert torch.equal(guard.protect(e0, seed=7), guard.protect(e0, seed=7))
    print("\n  fresh noise per call, reproducible under a seed: True")

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        p = guard.save(Path(tmp) / "remote_rag.pt")
        g2 = RemoteRag.load(p, device="cpu")
        assert torch.equal(guard.protect(e0, seed=3), g2.protect(e0, seed=3))
        print("  save/load round-trip reproduces the mechanism exactly: True")
