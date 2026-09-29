from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
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
class EguardConfig:
    backbone: str | None = None        
    hidden: int = 768                  
    num_layers: int = 6
    num_heads: int = 12
    num_pseudo_tokens: int = 8         
    ffn_mult: int = 4
    dropout: float = 0.1
    out_dim: int | None = None         
    pool: str = "mean"                
    stochastic: bool = False          
    bottleneck_weight: float = 1e-3   
    normalize_output: bool = True    

    alpha: float = 1.0                 
    mi_estimator: str = "infonce"      
    mi_hidden: int = 512
    mi_temperature: float = 0.07
    critic_steps: int = 1             
    critic_lr: float = 1e-4
    latent_model: str = "gte-base"     

    utility_loss: str = "structure"    
    utility_weight: float = 1.0
    structure_temperature: float = 0.05
    structure_kl_weight: float = 1.0   
    mnrl_scale: float = 20.0           
    mnrl_margin: float = 0.0           

    lr: float = 2e-5                   
    weight_decay: float = 0.0
    batch_size: int = 16               
    epochs: int = 10
    grad_clip: float = 1.0
    seed: int = 0
    device: str | None = None          

    def resolved_device(self) -> str:
        if self.device:
            return self.device
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"


class ProjectionNetwork(nn.Module):
    def __init__(self, dim: int, config: EguardConfig) -> None:
        super().__init__()
        self.dim = dim
        self.out_dim = config.out_dim or dim
        self.cfg = config

        if config.backbone:
            hidden, encoder = self._load_backbone(config)
        else:
            hidden, encoder = self._build_encoder(config)
        self.hidden = hidden
        self.encoder = encoder
        self.is_hf = bool(config.backbone)

        n_tok = config.num_pseudo_tokens
        self.n_tokens = n_tok
        self.expand = nn.Linear(dim, n_tok * hidden)
        self.pos = nn.Parameter(torch.zeros(1, n_tok, hidden))
        nn.init.normal_(self.pos, std=0.02)
        self.in_norm = nn.LayerNorm(hidden)
        self.out_norm = nn.LayerNorm(hidden)

        head_out = self.out_dim * (2 if config.stochastic else 1)
        self.head = nn.Linear(hidden, head_out)

    @staticmethod
    def _build_encoder(config: EguardConfig) -> tuple[int, nn.Module]:
        hidden = config.hidden
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=config.num_heads,
            dim_feedforward=hidden * config.ffn_mult,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        return hidden, nn.TransformerEncoder(layer, num_layers=config.num_layers)

    @staticmethod
    def _load_backbone(config: EguardConfig) -> tuple[int, nn.Module]:
        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise ImportError(
                f"backbone={config.backbone!r} needs transformers: pip install transformers"
            ) from exc
        model = AutoModel.from_pretrained(config.backbone)
        hidden = int(model.config.hidden_size)
        return hidden, model

    def _run_encoder(self, seq: torch.Tensor) -> torch.Tensor:
        if self.is_hf:
            mask = torch.ones(seq.shape[:2], dtype=torch.long, device=seq.device)
            return self.encoder(inputs_embeds=seq, attention_mask=mask).last_hidden_state
        return self.encoder(seq)

    def forward(self, e: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b = e.shape[0]
        seq = self.expand(e).view(b, self.n_tokens, self.hidden)
        seq = self.in_norm(seq + self.pos)
        seq = self._run_encoder(seq)
        pooled = seq[:, 0] if self.cfg.pool == "first" else seq.mean(dim=1)
        out = self.head(self.out_norm(pooled))

        kl = torch.zeros((), device=e.device, dtype=e.dtype)
        if self.cfg.stochastic:
            mu, logvar = out.chunk(2, dim=-1)
            logvar = logvar.clamp(-8.0, 8.0)
            out = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if self.training else mu
            kl = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar).sum(dim=-1).mean()

        if self.cfg.normalize_output:
            out = F.normalize(out, dim=-1)
        return out, kl


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
        nn.Linear(hidden, out_dim),
    )


class MIEstimator(nn.Module):
    def lower_bound(self, z: torch.Tensor, e_prime: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class InfoNCEEstimator(MIEstimator):
    def __init__(self, z_dim: int, e_dim: int, hidden: int = 512, temperature: float = 0.07) -> None:
        super().__init__()
        self.f = _mlp(z_dim, hidden, hidden)
        self.g = _mlp(e_dim, hidden, hidden)
        self.temperature = temperature

    def lower_bound(self, z: torch.Tensor, e_prime: torch.Tensor) -> torch.Tensor:
        a = F.normalize(self.f(z), dim=-1)
        b = F.normalize(self.g(e_prime), dim=-1)
        logits = (a @ b.t()) / self.temperature
        labels = torch.arange(z.shape[0], device=z.device)
        loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))
        return math.log(max(z.shape[0], 2)) - loss


class MineEstimator(MIEstimator):
    def __init__(self, z_dim: int, e_dim: int, hidden: int = 512) -> None:
        super().__init__()
        self.t = _mlp(z_dim + e_dim, hidden, 1)

    def lower_bound(self, z: torch.Tensor, e_prime: torch.Tensor) -> torch.Tensor:
        joint = self.t(torch.cat([z, e_prime], dim=-1)).mean()
        shuffled = e_prime[torch.randperm(e_prime.shape[0], device=e_prime.device)]
        marginal = self.t(torch.cat([z, shuffled], dim=-1)).squeeze(-1)
        return joint - (torch.logsumexp(marginal, dim=0) - math.log(marginal.shape[0]))


class ProbeEstimator(MIEstimator):
    def __init__(self, z_dim: int, e_dim: int, hidden: int = 512) -> None:
        super().__init__()
        self.h = _mlp(e_dim, hidden, z_dim)

    def lower_bound(self, z: torch.Tensor, e_prime: torch.Tensor) -> torch.Tensor:
        return F.cosine_similarity(self.h(e_prime), z, dim=-1).mean()


MI_ESTIMATORS: dict[str, type[MIEstimator]] = {
    "infonce": InfoNCEEstimator,
    "mine": MineEstimator,
    "probe": ProbeEstimator,
}


def multiple_negatives_ranking_loss(
    anchor: torch.Tensor, positive: torch.Tensor, *, scale: float = 20.0, margin: float = 0.0
) -> torch.Tensor:
    a = F.normalize(anchor, dim=-1)
    p = F.normalize(positive, dim=-1)
    scores = a @ p.t()
    if margin > 0.0:
        pos = scores.diagonal().unsqueeze(1)
        violation = (scores - pos + margin).clamp_min(0.0)
        n = scores.shape[0]
        violation = violation * (1.0 - torch.eye(n, device=scores.device))
        return violation.sum() / max(n, 1)
    labels = torch.arange(scores.shape[0], device=scores.device)
    return F.cross_entropy(scores * scale, labels)


def _offdiag(m: torch.Tensor) -> torch.Tensor:
    n = m.shape[0]
    keep = ~torch.eye(n, dtype=torch.bool, device=m.device)
    return m[keep].view(n, n - 1)


def structure_preservation_loss(
    e: torch.Tensor, e_prime: torch.Tensor, *, temperature: float = 0.05, kl_weight: float = 1.0
) -> torch.Tensor:
    a = F.normalize(e, dim=-1)
    b = F.normalize(e_prime, dim=-1)
    sim_src, sim_dst = _offdiag(a @ a.t()), _offdiag(b @ b.t())
    loss = F.mse_loss(sim_dst, sim_src)

    if kl_weight > 0.0 and sim_src.shape[1] > 1:
        p = F.softmax(sim_src / temperature, dim=-1)
        log_q = F.log_softmax(sim_dst / temperature, dim=-1)
        loss = loss + kl_weight * F.kl_div(log_q, p, reduction="batchmean")
    return loss


def _vectors_of(x: Any) -> np.ndarray:
    if EmbeddingSet is not None and isinstance(x, EmbeddingSet):
        return np.asarray(x.vectors)
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _to_tensor(x: Any, device: str | torch.device) -> torch.Tensor:
    arr = np.ascontiguousarray(_vectors_of(x), dtype=np.float32)
    return torch.from_numpy(arr).to(device)


class Eguard(nn.Module):
    def __init__(self, dim: int, config: EguardConfig | None = None, **overrides: Any) -> None:
        super().__init__()
        cfg = config or EguardConfig()
        for key, value in overrides.items():
            if not hasattr(cfg, key):
                raise TypeError(f"Unknown Eguard option {key!r}. Known: {sorted(vars(cfg))}")
            setattr(cfg, key, value)

        self.cfg = cfg
        self.dim = dim
        self.device = torch.device(cfg.resolved_device())
        torch.manual_seed(cfg.seed)

        self.projection = ProjectionNetwork(dim, cfg).to(self.device)
        self.critic: MIEstimator | None = None
        self.z_dim: int | None = None      # g_a's output width; recorded for reload
        self._latent_encoder: Any = None
        self.history: list[dict[str, float]] = []

    def _encode_latents(self, texts: Sequence[str]) -> np.ndarray:
        if self._latent_encoder is None:
            from models import get_model

            self._latent_encoder = get_model(self.cfg.latent_model, device=str(self.device))
        return np.asarray(self._latent_encoder.encode(list(texts), show_progress=True))

    def fit(
        self,
        embeddings: Any,
        texts: Sequence[str] | None = None,
        *,
        latents: np.ndarray | None = None,
        pairs: tuple[Any, Any] | None = None,
        epochs: int | None = None,
        verbose: bool = True,
    ) -> "Eguard":
        cfg = self.cfg
        e_all = _to_tensor(embeddings, self.device)
        n = e_all.shape[0]
        if e_all.shape[1] != self.dim:
            raise ValueError(f"expected embeddings of dim {self.dim}, got {e_all.shape[1]}")

        wants_mnrl = cfg.utility_loss in ("mnrl", "both")
        if wants_mnrl and pairs is None:
            raise ValueError(
                "utility_loss={!r} needs `pairs=(anchors, positives)`; use "
                "utility_loss='structure' for a corpus with no labelled pairs.".format(
                    cfg.utility_loss
                )
            )
        anchors = positives = None
        if wants_mnrl:
            anchors, positives = (_to_tensor(p, self.device) for p in pairs)  # type: ignore[misc]

        needs_mi = cfg.alpha != 0.0
        z_all: torch.Tensor | None = None
        if needs_mi:
            if latents is None:
                if texts is None:
                    raise ValueError(
                        "L_1 needs g_a(x): pass `texts` (e.g. selection.texts), pass "
                        "`latents=`, or set alpha=0 to train utility only."
                    )
                if len(texts) != n:
                    raise ValueError(f"texts/embeddings length mismatch: {len(texts)} vs {n}")
                latents = self._encode_latents(texts)
            z_all = _to_tensor(latents, self.device)
            if z_all.shape[0] != n:
                raise ValueError(f"latents/embeddings length mismatch: {z_all.shape[0]} vs {n}")
            if self.critic is None:
                self.z_dim = int(z_all.shape[1])
                self.critic = MI_ESTIMATORS[cfg.mi_estimator](
                    self.z_dim, self.projection.out_dim, hidden=cfg.mi_hidden,
                    **({"temperature": cfg.mi_temperature} if cfg.mi_estimator == "infonce" else {}),
                ).to(self.device)

        opt_p = torch.optim.Adam(
            self.projection.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        opt_c = (
            torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
            if self.critic is not None else None
        )

        # A batch is also the negative pool for both InfoNCE and the similarity
        # matrix, so a batch of 1 makes every loss degenerate.
        bs = max(2, min(cfg.batch_size, n))
        generator = torch.Generator(device="cpu").manual_seed(cfg.seed)
        total_epochs = epochs if epochs is not None else cfg.epochs

        for epoch in range(total_epochs):
            self.projection.train()
            perm = torch.randperm(n, generator=generator).to(self.device)
            sums = {"mi": 0.0, "utility": 0.0, "kl": 0.0, "total": 0.0}
            steps = 0

            for start in range(0, n - 1, bs):
                idx = perm[start : start + bs]
                if idx.numel() < 2:
                    continue
                e_b = e_all[idx]
                z_b = z_all[idx] if z_all is not None else None

                if self.critic is not None and opt_c is not None:
                    for _ in range(cfg.critic_steps):
                        with torch.no_grad():
                            e_prime_fixed, _ = self.projection(e_b)
                        opt_c.zero_grad(set_to_none=True)
                        (-self.critic.lower_bound(z_b, e_prime_fixed)).backward()
                        opt_c.step()

                opt_p.zero_grad(set_to_none=True)
                e_prime, kl = self.projection(e_b)

                mi = (
                    self.critic.lower_bound(z_b, e_prime)
                    if self.critic is not None
                    else torch.zeros((), device=self.device)
                )

                utility = torch.zeros((), device=self.device)
                if cfg.utility_loss in ("structure", "both"):
                    utility = utility + structure_preservation_loss(
                        e_b, e_prime,
                        temperature=cfg.structure_temperature,
                        kl_weight=cfg.structure_kl_weight,
                    )
                if wants_mnrl:
                    a_p, _ = self.projection(anchors[idx])  
                    p_p, _ = self.projection(positives[idx]) 
                    utility = utility + multiple_negatives_ranking_loss(
                        a_p, p_p, scale=cfg.mnrl_scale, margin=cfg.mnrl_margin
                    )

                total = cfg.alpha * mi + cfg.utility_weight * utility
                if cfg.stochastic:
                    total = total + cfg.bottleneck_weight * kl
                total.backward()
                if cfg.grad_clip:
                    nn.utils.clip_grad_norm_(self.projection.parameters(), cfg.grad_clip)
                opt_p.step()

                for key, value in (("mi", mi), ("utility", utility), ("kl", kl), ("total", total)):
                    sums[key] += float(value.detach())
                steps += 1

            row = {"epoch": epoch + 1, **{k: v / max(steps, 1) for k, v in sums.items()}}
            self.history.append(row)
            if verbose:
                print(
                    f"epoch {row['epoch']:>3}/{total_epochs}  "
                    f"MI(hat) {row['mi']:+.4f}  L2 {row['utility']:.4f}  "
                    f"KL {row['kl']:.4f}  total {row['total']:+.4f}"
                )

        self.projection.eval()
        return self

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        return self.projection(e)[0]

    @torch.no_grad()
    def protect(self, embeddings: Any, *, batch_size: int = 512) -> Any:
        self.projection.eval()
        e = _to_tensor(embeddings, self.device)
        out = torch.cat(
            [self.projection(e[i : i + batch_size])[0] for i in range(0, e.shape[0], batch_size)]
        )

        if EmbeddingSet is not None and isinstance(embeddings, EmbeddingSet):
            return EmbeddingSet(
                vectors=out.cpu().numpy().astype(embeddings.vectors.dtype, copy=False),
                ids=list(embeddings.ids),
                indices=list(embeddings.indices),
                model=embeddings.model,
                dataset=embeddings.dataset,
                meta={**embeddings.meta, "defense": "eguard", "eguard": asdict(self.cfg)},
            )
        if torch.is_tensor(embeddings):
            return out.to(embeddings.device, embeddings.dtype)
        return out.cpu().numpy()

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
        overlap = [
            len(set(top_a[i].tolist()) & set(top_b[i].tolist())) / kk for i in range(n)
        ]

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
                "z_dim": self.z_dim,
                "config": asdict(self.cfg),
                "projection": self.projection.state_dict(),
                "critic": self.critic.state_dict() if self.critic is not None else None,
                "history": self.history,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: str | Path, *, device: str | None = None) -> "Eguard":
        ckpt = torch.load(Path(path), map_location="cpu", weights_only=False)
        cfg = EguardConfig(**ckpt["config"])
        if device:
            cfg.device = device
        guard = cls(ckpt["dim"], cfg)
        guard.projection.load_state_dict(ckpt["projection"])
        guard.projection.eval()
        if ckpt.get("critic") is not None and ckpt.get("z_dim"):
            guard.z_dim = int(ckpt["z_dim"])
            guard.critic = MI_ESTIMATORS[cfg.mi_estimator](
                guard.z_dim, guard.projection.out_dim, hidden=cfg.mi_hidden,
                **({"temperature": cfg.mi_temperature} if cfg.mi_estimator == "infonce" else {}),
            ).to(guard.device)
            guard.critic.load_state_dict(ckpt["critic"])
        guard.history = ckpt.get("history", [])
        return guard

    def __repr__(self) -> str:
        return (
            f"Eguard(dim={self.dim}, out_dim={self.projection.out_dim}, "
            f"backbone={self.cfg.backbone or 'scratch'}, layers={self.cfg.num_layers}, "
            f"alpha={self.cfg.alpha}, mi={self.cfg.mi_estimator!r}, "
            f"utility={self.cfg.utility_loss!r}, stochastic={self.cfg.stochastic})"
        )

if __name__ == "__main__":  
    from dataloader import Record, Selection
    from models import get_model

    sel = Selection(
        dataset="synthetic",
        indices=list(range(256)),
        records=[Record(id=str(i), text=f"smoke test record number {i} about topic "
                                        f"{i % 8}") for i in range(256)],
    )
    emb = get_model("hash", cache_dir=None).embed(sel, use_cache=False, show_progress=False)
    print(emb)

    guard = Eguard(
        emb.dim,
        latent_model="hash",     
        num_layers=2, hidden=128, num_heads=4, num_pseudo_tokens=4,
        lr=1e-4, epochs=3, batch_size=32,
    )
    print(guard)
    guard.fit(emb, sel.texts)
    protected = guard.protect(emb)
    print(protected)
    print(json.dumps(guard.utility_report(emb, protected), indent=2))
