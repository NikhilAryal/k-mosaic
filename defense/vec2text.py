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
class Vec2TextConfig:


    noise_level: float = 0.01
    noise_scale: str = "absolute"

    normalize_input: bool = True
    renormalize_output: bool = False       

    victim_model: str | None = None        
    dataset: str | None = None             
    strict_dim_check: bool = True
    seed: int = 0
    device: str | None = None

    def __post_init__(self) -> None:
        if self.noise_scale not in ("absolute", "relative"):
            raise ValueError(
                f"noise_scale must be 'absolute' or 'relative', got {self.noise_scale!r}"
            )
        if self.noise_level < 0:
            raise ValueError(f"noise_level (lambda) must be >= 0, got {self.noise_level}")

    def resolved_device(self) -> str:
        if self.device:
            return self.device
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"


def noisy_embedding(
    e: torch.Tensor,
    noise_level: float,
    *,
    noise_scale: str = "absolute",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if noise_level <= 0:
        return e.clone()
    eps = torch.randn(e.shape, device=e.device, dtype=e.dtype, generator=generator)
    if noise_scale == "relative":
        scale = noise_level * e.norm(dim=-1, keepdim=True) / math.sqrt(e.shape[-1])
        return e + scale * eps
    return e + noise_level * eps

def _vectors_of(x: Any) -> np.ndarray:
    if EmbeddingSet is not None and isinstance(x, EmbeddingSet):
        return x.vectors
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _to_tensor(x: Any, device: str | torch.device) -> torch.Tensor:
    return torch.as_tensor(_vectors_of(x), dtype=torch.float32, device=device)


class Vec2TextDefense(nn.Module):
    def __init__(self, dim: int, config: Vec2TextConfig | None = None, **overrides: Any) -> None:
        super().__init__()
        cfg = config or Vec2TextConfig()
        for key, value in overrides.items():
            if not hasattr(cfg, key):
                raise TypeError(f"unknown Vec2TextConfig field {key!r}")
            setattr(cfg, key, value)
        cfg.__post_init__()

        self.cfg = cfg
        self.dim = int(dim)
        self.device = torch.device(cfg.resolved_device())
        self._fitted = False
        self.corpus_norm_mean: float = 1.0
        self.corpus_norm_std: float = 0.0
        self.to(self.device)

    def fit(self, embeddings: Any = None, *, verbose: bool = True) -> "Vec2TextDefense":
        if embeddings is not None:
            e = _to_tensor(embeddings, self.device)
            if e.shape[1] != self.dim:
                raise ValueError(f"embedding dim {e.shape[1]} != this defense's dim {self.dim}")
            norms = e.norm(dim=-1)
            self.corpus_norm_mean = float(norms.mean())
            self.corpus_norm_std = float(norms.std()) if norms.numel() > 1 else 0.0
            if verbose:
                print(f"[vec2text] corpus ||phi(x)||: mean={self.corpus_norm_mean:.4f} "
                      f"std={self.corpus_norm_std:.4f}  (n={len(norms)})")
        self._fitted = True

        report = self.privacy_report()
        if verbose:
            print(f"[vec2text] lambda={self.cfg.noise_level:g} -> noise norm "
                  f"{report['expected_noise_norm']:.4f}, SNR {report['snr']:.4f}")
        if report["snr"] < 1.0:
            warnings.warn(
                f"vec2text: at lambda={self.cfg.noise_level:g} the noise norm "
                f"({report['expected_noise_norm']:.3f}) exceeds the signal norm "
                f"({report['signal_norm']:.3f}). Table 7 shows retrieval collapsing in this "
                "regime (NDCG@10 0.302 -> 0.002 at lambda=0.1). Section 6's prose recommends "
                "0.1, but its own Table 7 puts the knee at 0.01 — see "
                "AMBIGUITIES['section6_lambda_typo'].",
                RuntimeWarning,
                stacklevel=2,
            )
        return self

    @torch.no_grad()
    def protect(
        self,
        embeddings: Any,
        *,
        noise_level: float | None = None,
        seed: int | None = None,
    ) -> Any:
        e = _to_tensor(embeddings, self.device)
        if self.cfg.strict_dim_check and e.shape[1] != self.dim:
            raise ValueError(
                f"defense was built for dim {self.dim} but got embeddings of dim {e.shape[1]}."
            )
        if self.cfg.normalize_input:
            e = F.normalize(e, dim=-1)

        lam = self.cfg.noise_level if noise_level is None else float(noise_level)
        gen = None
        if seed is not None:
            gen = torch.Generator(device=self.device).manual_seed(int(seed))

        out = noisy_embedding(e, lam, noise_scale=self.cfg.noise_scale, generator=gen)
        if self.cfg.renormalize_output:
            out = F.normalize(out, dim=-1)

        meta_defense = {
            **asdict(self.cfg),
            "noise_level_applied": lam,
            "noise_seed": seed,
        }
        if EmbeddingSet is not None and isinstance(embeddings, EmbeddingSet):
            return EmbeddingSet(
                vectors=out.cpu().numpy().astype(embeddings.vectors.dtype, copy=False),
                ids=list(embeddings.ids),
                indices=list(embeddings.indices),
                model=embeddings.model,
                dataset=embeddings.dataset,
                meta={**embeddings.meta, "defense": "vec2text", "vec2text": meta_defense},
            )
        if torch.is_tensor(embeddings):
            return out.to(embeddings.device, embeddings.dtype)
        return out.cpu().numpy()

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        return self.protect(e)

    @torch.no_grad()
    def privacy_report(self, noise_level: float | None = None) -> dict[str, Any]:

        lam = self.cfg.noise_level if noise_level is None else float(noise_level)
        signal = self.corpus_norm_mean if self.cfg.normalize_input is False else 1.0
        noise = (
            lam * signal if self.cfg.noise_scale == "relative"
            else lam * math.sqrt(self.dim)
        )
        return {
            "noise_level": lam,
            "noise_scale": self.cfg.noise_scale,
            "dim": self.dim,
            "signal_norm": signal,
            "corpus_norm_mean": self.corpus_norm_mean,
            "corpus_norm_std": self.corpus_norm_std,
            "expected_noise_norm": noise,
            "noise_to_signal": noise / signal if signal > 0 else float("inf"),
            "snr": signal / noise if noise > 0 else float("inf"),
            "renormalized": self.cfg.renormalize_output,
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
    def load(cls, path: str | Path, *, device: str | None = None) -> "Vec2TextDefense":
        ckpt = torch.load(Path(path), map_location="cpu", weights_only=False)
        cfg = Vec2TextConfig(**ckpt["config"])
        if device:
            cfg.device = device
        obj = cls(ckpt["dim"], cfg)
        obj.corpus_norm_mean = ckpt.get("corpus_norm_mean", 1.0)
        obj.corpus_norm_std = ckpt.get("corpus_norm_std", 0.0)
        obj._fitted = True
        return obj

    def __repr__(self) -> str:
        return (
            f"Vec2TextDefense(dim={self.dim}, lambda={self.cfg.noise_level:g}, "
            f"scale={self.cfg.noise_scale!r}, renorm={self.cfg.renormalize_output}, "
            f"snr={self.privacy_report()['snr']:.3f})"
        )


NoisyEmbedding = Vec2TextDefense

if __name__ == "__main__": 
    torch.manual_seed(0)
    d = 768

    e = F.normalize(torch.randn(256, d), dim=-1)
    g = torch.Generator().manual_seed(0)
    got = noisy_embedding(e, 0.01, generator=g)
    g2 = torch.Generator().manual_seed(0)
    want = e + 0.01 * torch.randn(e.shape, generator=g2)
    print(f"  matches `embeddings += noise_level * torch.randn(shape)`: "
          f"max err {float((got - want).abs().max()):.3e}")
    assert torch.allclose(got, want)
    for lam in (0.001, 0.01, 0.1, 1.0):
        measured = float((noisy_embedding(e, lam) - e).norm(dim=-1).mean())
        predicted = lam * math.sqrt(d)
        print(f"  lambda={lam:<6} measured noise norm {measured:8.4f}  "
              f"predicted lambda*sqrt(n) {predicted:8.4f}")
        assert abs(measured - predicted) / predicted < 0.01
    plain = Vec2TextDefense(d, noise_level=0.1, device="cpu")
    renorm = Vec2TextDefense(d, noise_level=0.1, renormalize_output=True, device="cpu")
    a, b = plain.protect(e, seed=1), renorm.protect(e, seed=1)
    assert torch.allclose(F.normalize(a, dim=-1), F.normalize(b, dim=-1), atol=1e-5)
    ua, ub = plain.utility_report(e, a), renorm.utility_report(e, b)
    assert all(abs(ua[k] - ub[k]) < 1e-6 for k in ua)
    print(f"\n  ||e'|| without renorm: mean {float(a.norm(dim=-1).mean()):.4f} "
          f"std {float(a.norm(dim=-1).std()):.4f}   with renorm: "
          f"{float(b.norm(dim=-1).mean()):.4f}")
    print("  directions identical, utility_report identical to 1e-6: True")
    print("  => renorm is a no-op for the victim; it only changes what the attacker sees")

    print("\n" + "=" * 78)
    rng = np.random.default_rng(0)
    latent = rng.normal(size=(400, 24))
    base = latent @ rng.normal(size=(24, d)) + 0.3 * rng.normal(size=(400, d))
    base = (base / np.linalg.norm(base, axis=1, keepdims=True)).astype("float32")
    e0 = torch.as_tensor(base)

    guard = Vec2TextDefense(d, device="cpu").fit(e0, verbose=False)
    paper_ndcg = {0.0: 0.302, 0.001: 0.302, 0.01: 0.296, 0.1: 0.002, 1.0: 0.001}
    print(f"{'lambda':>8} {'noise/signal':>13} {'recall@10':>10} {'spearman':>9} "
          f"{'paper NDCG@10':>14}")
    for lam in (0.0, 0.001, 0.01, 0.1, 1.0):
        rep = guard.privacy_report(noise_level=lam)
        util = guard.utility_report(e0, guard.protect(e0, noise_level=lam, seed=0), k=10)
        print(f"{lam:>8} {rep['noise_to_signal']:>13.4f} {util['recall_at_10']:>10.3f} "
              f"{util['sim_spearman']:>9.3f} {paper_ndcg[lam]:>14.3f}")

    assert not torch.equal(guard.protect(e0), guard.protect(e0))
    assert torch.equal(guard.protect(e0, seed=7), guard.protect(e0, seed=7))
    assert torch.equal(guard.protect(e0, noise_level=0.0), F.normalize(e0, dim=-1))
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        p = guard.save(Path(tmp) / "vec2text.pt")
        g2 = Vec2TextDefense.load(p, device="cpu")
        assert torch.equal(guard.protect(e0, seed=3), g2.protect(e0, seed=3))
        print("  save/load round-trip reproduces the mechanism exactly: True")
