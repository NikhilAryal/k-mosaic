from __future__ import annotations

import json
import math
import re
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:  
    from models import EmbeddingSet
except Exception:  
    EmbeddingSet = None 



@dataclass
class SparseConfig:

    concept_name: str = "entities"
    concept_tokens: tuple[str, ...] = ()      
    concept_entity_types: tuple[str, ...] = () 
    spacy_model: str = "en_core_web_sm"       
    removal: str = "delete"                   
    placeholder_token: str = "[MASK]"

    zeta: float = 1.1           
    gamma: float = -0.1                        
    log_alpha_init: float = 0.0                
    beta_init: float = 2.0 / 3.0               
    learn_beta: bool = True                    
    beta_min: float = 0.05                     
    lam: float = 1e-3                 
    l0_sign: str = "standard"                
    classifier_hidden: tuple[int, ...] = (256, 128)   
    lr: float = 1e-4                           
    mask_lr: float | None = 1e-2               
    epochs: int = 100                          
    batch_size: int = 64                       
    grad_clip: float = 0.0                     
    val_fraction: float = 0.1                  

    epsilon: float = 10.0                      
    epsilon_scale: str = "absolute"            
    delta_psd: float = 1e-6                    
    normalize_trace: bool = True               
    gamma_shape: int | None = None             
    normalize_input: bool = True
    renormalize_output: bool = False

    victim_model: str | None = None            
    dataset: str | None = None                 
    strict_dim_check: bool = True
    seed: int = 0
    device: str | None = None

    def __post_init__(self) -> None:
        if self.l0_sign not in ("standard", "paper"):
            raise ValueError(f"l0_sign must be 'standard' or 'paper', got {self.l0_sign!r}")
        if self.epsilon_scale not in ("absolute", "per_dim"):
            raise ValueError(
                f"epsilon_scale must be 'absolute' or 'per_dim', got {self.epsilon_scale!r}"
            )
        if self.removal not in ("delete", "placeholder", "unk"):
            raise ValueError(f"removal must be delete|placeholder|unk, got {self.removal!r}")
        self.concept_tokens = tuple(self.concept_tokens)
        self.concept_entity_types = tuple(self.concept_entity_types)
        self.classifier_hidden = tuple(self.classifier_hidden)

    def resolved_device(self) -> str:
        if self.device:
            return self.device
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"


_WS = re.compile(r"\s+")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?%)\]}])")


def _tidy(text: str) -> str:
    return _SPACE_BEFORE_PUNCT.sub(r"\1", _WS.sub(" ", text)).strip()


class ConceptExtractor:
    def __init__(self, config: SparseConfig, custom: Callable[[str], Sequence[str]] | None = None):
        self.cfg = config
        self.custom = custom
        self._nlp: Any = None
        self._token_re = self._compile_tokens(config.concept_tokens)

    @staticmethod
    def _compile_tokens(tokens: Sequence[str]) -> re.Pattern | None:
        if not tokens:
            return None
        alt = "|".join(sorted((re.escape(t) for t in tokens), key=len, reverse=True))
        return re.compile(rf"(?<!\w)(?:{alt})(?!\w)", re.IGNORECASE)

    @property
    def nlp(self) -> Any:
        if self._nlp is None:
            try:
                import spacy
            except ImportError as exc:  
                raise ImportError(
                    "concept_entity_types needs spaCy: pip install spacy && "
                    f"python -m spacy download {self.cfg.spacy_model}"
                ) from exc
            try:
                self._nlp = spacy.load(self.cfg.spacy_model)
            except OSError as exc: 
                raise OSError(
                    f"spaCy model {self.cfg.spacy_model!r} is not installed: "
                    f"python -m spacy download {self.cfg.spacy_model}"
                ) from exc
        return self._nlp

    def spans(self, text: str) -> list[tuple[int, int]]:
        found: list[tuple[int, int]] = []
        if self._token_re is not None:
            found += [m.span() for m in self._token_re.finditer(text)]
        if self.cfg.concept_entity_types:
            wanted = set(self.cfg.concept_entity_types)
            found += [
                (ent.start_char, ent.end_char)
                for ent in self.nlp(text).ents
                if ent.label_ in wanted
            ]
        if self.custom is not None:
            for tok in self.custom(text):
                pat = self._compile_tokens([tok])
                if pat is not None:
                    found += [m.span() for m in pat.finditer(text)]

        found.sort()
        merged: list[tuple[int, int]] = []
        for lo, hi in found:
            if merged and lo <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
            else:
                merged.append((lo, hi))
        return merged

    def tokens(self, text: str) -> list[str]:
        return [text[lo:hi] for lo, hi in self.spans(text)]

    def remove(self, text: str) -> str:
        spans = self.spans(text)
        if not spans:
            return text
        out, prev = [], 0
        for lo, hi in spans:
            out.append(text[prev:lo])
            if self.cfg.removal == "placeholder":
                out.append(self.cfg.placeholder_token)
            elif self.cfg.removal == "unk":
                out.append("unk")
            prev = hi
        out.append(text[prev:])
        return _tidy("".join(out))


@dataclass
class ConceptPairs:

    positives: list[str]
    negatives: list[str]
    concept_tokens: list[list[str]]
    indices: list[int]      # row in the input `texts` each pair came from
    n_dropped: int          # sentences with no concept token, excluded
    n_input: int

    def __len__(self) -> int:
        return len(self.positives)

    def summary(self) -> dict[str, Any]:
        vocab = sorted({t.lower() for toks in self.concept_tokens for t in toks})
        return {
            "n_input": self.n_input,
            "n_pairs": len(self.positives),
            "n_dropped_no_concept": self.n_dropped,
            "n_unique_concept_tokens": len(vocab),
            "example_concept_tokens": vocab[:20],
        }


def build_concept_pairs(
    texts: Sequence[str],
    extractor: ConceptExtractor,
    *,
    verbose: bool = True,
) -> ConceptPairs:
    pos, neg, toks, idx, dropped = [], [], [], [], 0
    for row, text in enumerate(texts):
        found = extractor.tokens(text)
        if not found:
            dropped += 1
            continue
        removed = extractor.remove(text)
        if removed.strip() == text.strip():
            dropped += 1
            continue
        pos.append(text)
        neg.append(removed)
        toks.append(found)
        idx.append(row)

    pairs = ConceptPairs(pos, neg, toks, idx, dropped, len(texts))
    if verbose:
        print(f"[sparse] concept pairs: {json.dumps(pairs.summary(), ensure_ascii=False)}")
    if not pos:
        raise ValueError(
            "No sentence contained a concept token — D+ is empty. Check "
            "concept_tokens / concept_entity_types against the corpus."
        )
    return pairs

class HardConcreteMask(nn.Module):
    def __init__(self, dim: int, config: SparseConfig) -> None:
        super().__init__()
        self.dim = dim
        self.cfg = config
        self.log_alpha = nn.Parameter(torch.full((dim,), float(config.log_alpha_init)))
        log_beta = torch.full((dim,), math.log(config.beta_init))
        self.log_beta = nn.Parameter(log_beta, requires_grad=bool(config.learn_beta))

    @property
    def beta(self) -> torch.Tensor:
        return self.log_beta.exp().clamp_min(self.cfg.beta_min)

    def forward(self, batch: int = 1) -> torch.Tensor:
        if not self.training:
            return self.deterministic().expand(batch, -1)
        u = torch.rand(batch, self.dim, device=self.log_alpha.device).clamp(1e-6, 1 - 1e-6)
        s = torch.sigmoid((torch.log(u) - torch.log1p(-u) + self.log_alpha) / self.beta)
        return self._stretch(s)

    def deterministic(self) -> torch.Tensor:
        return self._stretch(torch.sigmoid(self.log_alpha)).unsqueeze(0)

    def _stretch(self, s: torch.Tensor) -> torch.Tensor:
        return (s * (self.cfg.zeta - self.cfg.gamma) + self.cfg.gamma).clamp(0.0, 1.0)

    def l0_penalty(self) -> torch.Tensor:
        shift = math.log(-self.cfg.gamma / self.cfg.zeta)
        expected = torch.sigmoid(self.log_alpha - self.beta * shift).mean()
        return -expected if self.cfg.l0_sign == "paper" else expected

    @torch.no_grad()
    def active(self, threshold: float = 0.5) -> torch.Tensor:
        return (self.deterministic().squeeze(0) > threshold)


def _mlp(in_dim: int, hidden: Sequence[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)


def mahalanobis_noise(
    n_rows: int,
    sigma_diag: torch.Tensor,
    epsilon: float,
    *,
    shape: int | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if epsilon <= 0:
        raise ValueError(f"epsilon must be > 0, got {epsilon}")
    device, dtype = sigma_diag.device, sigma_diag.dtype
    n = int(sigma_diag.numel())
    k = float(shape if shape is not None else n)

    direction = torch.randn(n_rows, n, device=device, dtype=dtype, generator=generator)
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

    return radius.unsqueeze(-1) * sigma_diag.sqrt().unsqueeze(0) * direction


def sigma_from_mask(
    mask: torch.Tensor,
    *,
    delta_psd: float = 1e-6,
    normalize_trace: bool = True,
) -> torch.Tensor:
    m = mask.detach().clone().float().clamp_min(0.0)
    n = m.numel()
    if normalize_trace:
        total = float(m.sum())
        if total <= 1e-8:
            warnings.warn(
                "SPARSE: the learned mask is all-zero — no dimension was found "
                "privacy-sensitive. Falling back to an isotropic Σ. "
                "Lower `lam` and refit.",
                RuntimeWarning,
                stacklevel=2,
            )
            m = torch.ones_like(m)
        else:
            m = m * (n / total)
    return m + delta_psd

def _vectors_of(x: Any) -> np.ndarray:
    if EmbeddingSet is not None and isinstance(x, EmbeddingSet):
        return x.vectors
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _to_tensor(x: Any, device: str | torch.device) -> torch.Tensor:
    return torch.as_tensor(_vectors_of(x), dtype=torch.float32, device=device)


class Sparse(nn.Module):
    def __init__(self, dim: int, config: SparseConfig | None = None, **overrides: Any) -> None:
        super().__init__()
        cfg = config or SparseConfig()
        for key, value in overrides.items():
            if not hasattr(cfg, key):
                raise TypeError(f"Unknown Sparse option {key!r}. Known: {sorted(vars(cfg))}")
            setattr(cfg, key, value)
        cfg.__post_init__()

        self.cfg = cfg
        self.dim = dim
        self.device = torch.device(cfg.resolved_device())

        _cpu_state = torch.get_rng_state()
        _cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            torch.manual_seed(cfg.seed)
            self.gate = HardConcreteMask(dim, cfg).to(self.device)
        finally:
            torch.set_rng_state(_cpu_state)
            if _cuda_states is not None:
                torch.cuda.set_rng_state_all(_cuda_states)
        self.classifier: nn.Module | None = None
        self.history: list[dict[str, float]] = []
        self.pair_summary: dict[str, Any] = {}
        self._encoder: Any = None
        self._fitted = False
        self.register_buffer("_sigma", torch.ones(dim, device=self.device), persistent=False)
        self._refresh_sigma()

    @classmethod
    def from_mask(
        cls, mask: Any, config: SparseConfig | None = None, **overrides: Any
    ) -> "Sparse":
        m = torch.as_tensor(np.asarray(mask), dtype=torch.float32).flatten()
        obj = cls(int(m.numel()), config, **overrides)
        s = ((m.to(obj.device) - obj.cfg.gamma) / (obj.cfg.zeta - obj.cfg.gamma)).clamp(1e-6, 1 - 1e-6)
        with torch.no_grad():
            obj.gate.log_alpha.copy_(torch.log(s) - torch.log1p(-s))
        obj._fitted = True
        obj._refresh_sigma()
        return obj


    @property
    def mask(self) -> torch.Tensor:
        return self.gate.deterministic().squeeze(0).detach()

    @property
    def sigma_diag(self) -> torch.Tensor:
        return self._sigma

    def effective_epsilon(self, epsilon: float | None = None) -> float:
        eps = self.cfg.epsilon if epsilon is None else epsilon
        return eps * self.dim if self.cfg.epsilon_scale == "per_dim" else eps

    def _refresh_sigma(self) -> None:
        self._sigma = sigma_from_mask(
            self.mask,
            delta_psd=self.cfg.delta_psd,
            normalize_trace=self.cfg.normalize_trace,
        ).to(self.device)

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        if self._encoder is None:
            if self.cfg.victim_model is None:
                raise ValueError(
                    "fit() needs to embed D-. Either set victim_model='gtr-base' (the "
                    "registry name of phi), or pass encoder=/neg_embeddings= to fit()."
                )
            from models import get_model

            self._encoder = get_model(self.cfg.victim_model, device=str(self.device))
        return np.asarray(self._encoder.encode(list(texts), show_progress=True))

    def fit(
        self,
        embeddings: Any = None,
        texts: Sequence[str] | None = None,
        *,
        pairs: ConceptPairs | None = None,
        pos_embeddings: Any = None,
        neg_embeddings: Any = None,
        encoder: Any = None,
        concept_fn: Callable[[str], Sequence[str]] | None = None,
        epochs: int | None = None,
        verbose: bool = True,
    ) -> "Sparse":
        cfg = self.cfg
        if encoder is not None:
            self._encoder = encoder

        if pos_embeddings is None or neg_embeddings is None:
            if pairs is None:
                if texts is None:
                    raise ValueError("fit() needs `texts`, or `pairs`, or both embedding halves.")
                extractor = ConceptExtractor(cfg, custom=concept_fn)
                pairs = build_concept_pairs(texts, extractor, verbose=verbose)

            self.pair_summary = pairs.summary()
            if pos_embeddings is None:
                rows = _vectors_of(embeddings) if embeddings is not None else None
                if rows is not None and len(rows) == pairs.n_input:
                    pos_embeddings = rows[np.asarray(pairs.indices, dtype=int)]
                else:
                    if rows is not None:
                        warnings.warn(
                            f"SPARSE: `embeddings` has {len(rows)} rows but the pairs were built "
                            f"from {pairs.n_input} texts, so rows cannot be aligned. Re-encoding "
                            "D+ with Phi instead. Pass embeddings and texts of equal length to "
                            "avoid the extra encode.",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                    pos_embeddings = self._encode(pairs.positives)
            if neg_embeddings is None:
                neg_embeddings = self._encode(pairs.negatives)

        pos = _to_tensor(pos_embeddings, self.device)
        neg = _to_tensor(neg_embeddings, self.device)
        if pos.shape != neg.shape:
            raise ValueError(f"D+/D- shape mismatch: {tuple(pos.shape)} vs {tuple(neg.shape)}")
        if pos.shape[1] != self.dim:
            raise ValueError(f"embedding dim {pos.shape[1]} != this Sparse's dim {self.dim}")
        if cfg.normalize_input:
            pos, neg = F.normalize(pos, dim=-1), F.normalize(neg, dim=-1)

        g = torch.Generator(device="cpu").manual_seed(cfg.seed)
        perm = torch.randperm(pos.shape[0], generator=g)
        n_val = int(cfg.val_fraction * len(perm)) if cfg.val_fraction > 0 else 0
        val_idx, train_idx = perm[:n_val].to(self.device), perm[n_val:].to(self.device)

        self.classifier = _mlp(self.dim, cfg.classifier_hidden).to(self.device)
        params = list(self.gate.parameters()) + list(self.classifier.parameters())

        opt = torch.optim.Adam(
            [
                {"params": list(self.classifier.parameters()), "lr": cfg.lr},
                {"params": list(self.gate.parameters()),
                 "lr": cfg.lr if cfg.mask_lr is None else cfg.mask_lr},
            ]
        )

        n_epochs = epochs if epochs is not None else cfg.epochs
        self.history = []
        for epoch in range(n_epochs):
            self.gate.train()
            self.classifier.train()
            order = train_idx[torch.randperm(len(train_idx), device=self.device)]
            agg = {"loss": 0.0, "cls": 0.0, "reg": 0.0, "acc": 0.0, "n": 0.0}

            for start in range(0, len(order), cfg.batch_size):
                idx = order[start : start + cfg.batch_size]
                p, q = pos[idx], neg[idx]
                b = p.shape[0]
                m = self.gate(b)
                logits = torch.cat([self.classifier(p * m), self.classifier(q * m)]).squeeze(-1)
                target = torch.cat([torch.ones(b, device=self.device),
                                    torch.zeros(b, device=self.device)])
                l_cls = F.binary_cross_entropy_with_logits(logits, target)
                l_reg = self.gate.l0_penalty()
                loss = l_cls + cfg.lam * l_reg

                opt.zero_grad(set_to_none=True)
                loss.backward()
                if cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
                opt.step()

                with torch.no_grad():
                    acc = ((logits > 0).float() == target).float().mean()
                agg["loss"] += float(loss.detach()) * b
                agg["cls"] += float(l_cls.detach()) * b
                agg["reg"] += float(l_reg.detach()) * b
                agg["acc"] += float(acc) * b
                agg["n"] += b

            n = max(agg.pop("n"), 1.0)
            row = {k: v / n for k, v in agg.items()}
            row["epoch"] = float(epoch)
            row["active"] = float(self.gate.active().sum())
            if n_val:
                row["val_acc"] = self._separability(pos[val_idx], neg[val_idx])
            self.history.append(row)

            if verbose and (epoch % max(1, n_epochs // 10) == 0 or epoch == n_epochs - 1):
                extra = f" val_acc {row['val_acc']:.3f}" if n_val else ""
                print(
                    f"[sparse] epoch {epoch + 1:3d}/{n_epochs}  loss {row['loss']:.4f}  "
                    f"cls {row['cls']:.4f}  L0 {row['reg']:.4f}  acc {row['acc']:.3f}{extra}  "
                    f"active {int(row['active'])}/{self.dim}"
                )

        self.gate.eval()
        self.classifier.eval()
        self._fitted = True
        self._refresh_sigma()

        if int(self.gate.active().sum()) == 0:
            warnings.warn(
                f"SPARSE: no dimension survived at lam={cfg.lam}. The mask is empty and sigma will "
                "fall back to isotropic. Lower `lam` or check that D+/D- are "
                "actually separable.",
                RuntimeWarning,
                stacklevel=2,
            )
        return self

    @torch.no_grad()
    def _separability(self, pos: torch.Tensor, neg: torch.Tensor) -> float:
        if self.classifier is None or pos.numel() == 0:
            return float("nan")
        was = self.gate.training
        self.gate.eval()
        m = self.gate.deterministic()
        logits = torch.cat([self.classifier(pos * m), self.classifier(neg * m)]).squeeze(-1)
        target = torch.cat([torch.ones(len(pos), device=self.device),
                            torch.zeros(len(neg), device=self.device)])
        self.gate.train(was)
        return float(((logits > 0).float() == target).float().mean())

    @torch.no_grad()
    def protect(
        self,
        embeddings: Any,
        *,
        epsilon: float | None = None,
        batch_size: int = 4096,
        seed: int | None = None,
    ) -> Any:

        if not self._fitted:
            warnings.warn(
                "SPARSE: protect() called before fit(); the mask is still at its "
                "initialisation, so this is anisotropic noise in arbitrary directions.",
                RuntimeWarning,
                stacklevel=2,
            )
        eps = self.effective_epsilon(epsilon)
        e = _to_tensor(embeddings, self.device)
        if self.cfg.strict_dim_check and e.shape[1] != self.dim:
            raise ValueError(
                f"mask was fit for dim {self.dim} but got embeddings of dim {e.shape[1]}. "
                "A mask indexes one specific encoder's coordinates and does not transfer."
            )
        if self.cfg.normalize_input:
            e = F.normalize(e, dim=-1)

        gen = None
        if seed is not None:
            gen = torch.Generator(device=self.device).manual_seed(int(seed))

        sigma = self.sigma_diag.to(self.device)
        chunks = [
            e[i : i + batch_size]
            + mahalanobis_noise(
                e[i : i + batch_size].shape[0], sigma, eps,
                shape=self.cfg.gamma_shape, generator=gen,
            )
            for i in range(0, e.shape[0], batch_size)
        ]
        out = torch.cat(chunks) if chunks else e
        if self.cfg.renormalize_output:
            out = F.normalize(out, dim=-1)

        meta_defense = {
            **asdict(self.cfg),
            "epsilon_requested": self.cfg.epsilon if epsilon is None else epsilon,
            "epsilon_effective": eps,
            "noise_seed": seed,
        }
        if EmbeddingSet is not None and isinstance(embeddings, EmbeddingSet):
            return EmbeddingSet(
                vectors=out.cpu().numpy().astype(embeddings.vectors.dtype, copy=False),
                ids=list(embeddings.ids),
                indices=list(embeddings.indices),
                model=embeddings.model,
                dataset=embeddings.dataset,
                meta={**embeddings.meta, "defense": "sparse", "sparse": meta_defense},
            )
        if torch.is_tensor(embeddings):
            return out.to(embeddings.device, embeddings.dtype)
        return out.cpu().numpy()

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        return self.protect(e)

    @torch.no_grad()
    def privacy_report(self, epsilon: float | None = None) -> dict[str, Any]:
        requested = self.cfg.epsilon if epsilon is None else epsilon
        eps = self.effective_epsilon(epsilon)
        m = self.mask.cpu()
        sigma = self.sigma_diag.cpu()
        c = float(sigma.min())
        active = int((m > 0.5).sum())
        n = self.dim
        e_y2 = (n * (n + 1)) / (eps**2)
        expected_sq_norm = e_y2 * float(sigma.mean())
        return {
            "epsilon": requested,
            "epsilon_scale": self.cfg.epsilon_scale,
            "epsilon_effective": eps,
            "dim": n,
            "active_dims": active,
            "active_fraction": active / n,
            "mask_mean": float(m.mean()),
            "sigma_min": c,
            "sigma_max": float(sigma.max()),
            "sigma_trace": float(sigma.sum()),
            "euclidean_lower_constant_1_over_sqrt_n": 1.0 / math.sqrt(n),
            "euclidean_upper_constant_1_over_sqrt_c": 1.0 / math.sqrt(max(c, 1e-30)),
            "expected_noise_norm": math.sqrt(expected_sq_norm),
            "expected_noise_norm_lapmech": math.sqrt(e_y2),
            "snr_vs_unit_norm": 1.0 / max(math.sqrt(expected_sq_norm), 1e-12),
            "top_sensitive_dims": torch.argsort(m, descending=True)[:20].tolist(),
            "separability_val_acc": self.history[-1].get("val_acc") if self.history else None,
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
                "gate": self.gate.state_dict(),
                "classifier": self.classifier.state_dict() if self.classifier is not None else None,
                "mask": self.mask.cpu(),
                "history": self.history,
                "pair_summary": self.pair_summary,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: str | Path, *, device: str | None = None) -> "Sparse":
        ckpt = torch.load(Path(path), map_location="cpu", weights_only=False)
        cfg = SparseConfig(**ckpt["config"])
        if device:
            cfg.device = device
        obj = cls(ckpt["dim"], cfg)
        obj.gate.load_state_dict(ckpt["gate"])
        obj.gate.eval()
        if ckpt.get("classifier") is not None:
            obj.classifier = _mlp(obj.dim, cfg.classifier_hidden).to(obj.device)
            obj.classifier.load_state_dict(ckpt["classifier"])
            obj.classifier.eval()
        obj.history = ckpt.get("history", [])
        obj.pair_summary = ckpt.get("pair_summary", {})
        obj._fitted = True
        obj._refresh_sigma()
        return obj

    def __repr__(self) -> str:
        active = int((self.mask > 0.5).sum()) if self._fitted else -1
        return (
            f"Sparse(dim={self.dim}, concept={self.cfg.concept_name!r}, eps={self.cfg.epsilon}, "
            f"lam={self.cfg.lam}, active={active}/{self.dim}, "
            f"fitted={self._fitted}, l0_sign={self.cfg.l0_sign!r})"
        )


if __name__ == "__main__": 
    print("\n" + "=" * 78)
    print("Smoke test on synthetic embeddings")
    print("=" * 78)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    n, d, sensitive = 512, 64, [3, 11, 27, 40]
    base = rng.normal(size=(n, d)).astype("float32")
    pos, neg = base.copy(), base.copy()
    pos[:, sensitive] += 3.0
    neg[:, sensitive] -= 3.0

    sp = Sparse(d, epsilon=10.0, lam=1e-3, epochs=60, batch_size=64, device="cpu", seed=0)
    sp.fit(pos_embeddings=pos, neg_embeddings=neg, verbose=True)

    found = torch.argsort(sp.mask, descending=True)[: len(sensitive)].tolist()
    print(f"\nplanted sensitive dims : {sorted(sensitive)}")
    print(f"top-{len(sensitive)} dims by mask   : {sorted(found)}")
    print(f"recovered              : {sorted(found) == sorted(sensitive)}")
    print(f"\nprivacy_report: {json.dumps(sp.privacy_report(), indent=2, default=str)}")

    e = torch.as_tensor(base)
    e_prime = sp.protect(e, seed=0)
    print(f"utility_report: {json.dumps(sp.utility_report(e, e_prime), indent=2)}")

    iso = Sparse.from_mask(np.ones(d), epsilon=10.0, device="cpu")
    z = mahalanobis_noise(4096, iso.sigma_diag, 10.0)
    print(
        f"\nisotropic check: mean‖Z‖ = {float(z.norm(dim=-1).mean()):.3f}  "
        f"vs Gamma(n,1/eps) mean = {d / 10.0:.3f}  (equal => reduces to LapMech)"
    )
