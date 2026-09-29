from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F


def _renorm(X: torch.Tensor) -> torch.Tensor:
    return X / torch.norm(X, p=2, dim=1, keepdim=True).clamp_min(1e-12)


def insert_gaussian_noise(X: torch.Tensor, noise_level: float) -> torch.Tensor:
    return _renorm(X + noise_level * torch.randn(X.shape, device=X.device))


def dp_gaussian_embeddings(
    X: torch.Tensor, epsilon: float = 1.0, delta: float = 1e-5, sensitivity: float = 2.0
) -> torch.Tensor:
    sigma = (math.sqrt(2 * math.log(1.25 / delta)) * sensitivity) / epsilon
    noise = torch.normal(0.0, sigma, X.shape, device=X.device)
    return _renorm(X + sigma * noise)

def _angular_density(x: float, d: int, eps: float) -> float:
    return math.exp(-eps * x) * math.pow(math.sin(x), d - 2)


def _pur_arc(d: int, eps: float) -> float:
    from scipy import integrate

    a, b, u = 0.0, math.pi, np.random.uniform(0, 1)
    denom = integrate.quad(_angular_density, 0, math.pi, args=(d, eps))[0]
    theta = b / 2
    for _ in range(24):
        theta = (a + b) / 2
        y = integrate.quad(_angular_density, 0, theta, args=(d, eps))[0] / denom
        if y < u:
            a = theta
        else:
            b = theta
    return theta


def pur_mech(X: torch.Tensor, eps: float) -> torch.Tensor:
    X = F.normalize(X, dim=-1)
    n, d = X.shape
    direct = torch.randn((n, d), device=X.device)
    direct = direct - torch.sum(direct * X, dim=-1, keepdim=True) * X  # tangent component
    direct = F.normalize(direct, dim=-1)
    theta = torch.tensor(
        [[_pur_arc(d, eps)] * d for _ in range(n)], dtype=torch.float, device=X.device
    )
    return torch.cos(theta) * X + torch.sin(theta) * direct


def lap_mech(X: torch.Tensor, eps: float) -> torch.Tensor:
    X = F.normalize(X, dim=-1)
    n, d = X.shape
    scale = torch.distributions.gamma.Gamma(d, eps).sample((n,)).to(X.device)
    noise = F.normalize(torch.randn((n, d), device=X.device), dim=-1)
    return X + scale.unsqueeze(-1) * noise


def shuffle_embeddings(X: torch.Tensor, perm: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    if perm is None:
        perm = torch.randperm(X.shape[1], device=X.device)
    return _renorm(X[:, perm]), perm


def mask_embeddings(X: torch.Tensor) -> torch.Tensor:
    X_mask = X.clone()
    X_mask[:, 0] = 1.0
    return _renorm(X_mask)


def _wet_first_row(n: int, k: int, rng) -> torch.Tensor:
    row = [0.0] * n
    for position in rng.sample(range(n), k=k):
        row[position] = rng.random()
    return torch.FloatTensor(row).reshape(n)


def _is_full_rank_circulant(first_row: torch.Tensor) -> bool:
    return bool(np.all(np.abs(np.fft.fft(first_row.numpy())) > 1e-10))


def wet_transform_matrix(dim: int, seed: int = 42) -> torch.Tensor:
    import random as _random

    rng = _random.Random(seed)
    first_row = _wet_first_row(dim, dim, rng)
    if not _is_full_rank_circulant(first_row):
        raise RuntimeError("WET: drew a rank-deficient circulant row; retry with another seed")

    rows, curr = [], first_row.clone()
    for i in range(dim):
        values = curr.clone()
        values /= torch.sum(values)
        rows.append(values)
        curr = torch.roll(curr, 1)
        if curr.equal(first_row) and i + 1 < dim:
            first_row = _wet_first_row(dim, dim, rng)
            if not _is_full_rank_circulant(first_row):
                raise RuntimeError("WET: rank-deficient circulant row on retry")
            curr = first_row.clone()
    return torch.stack(rows)


def wet_transform(X: torch.Tensor, T: torch.Tensor | None = None, seed: int = 42) -> tuple[torch.Tensor, torch.Tensor]:
    if T is None:
        T = wet_transform_matrix(X.shape[1], seed=seed)
    T = T.to(X.device, X.dtype)
    return _renorm(X @ T.T), T


def eguard_transform(
    X: torch.Tensor,
    guard: "Any" = None,
    checkpoint: "str | Path | None" = None,
) -> tuple[torch.Tensor, "Any"]:
    if guard is None:
        if checkpoint is None:
            raise ValueError(
                "defense 'eguard' needs a trained projection network: pass "
                "eguard_checkpoint=<path to the .pt written by train_algen.py "
                "--stages eguard>."
            )
        from defense import Eguard

        guard = Eguard.load(checkpoint, device=str(X.device))
    return _renorm(guard.protect(X)), guard


def sparse_transform(
    X: torch.Tensor,
    defender: "Any" = None,
    checkpoint: "str | Path | None" = None,
    epsilon: float = 10.0,
) -> tuple[torch.Tensor, "Any"]:
    if defender is None:
        if checkpoint is None:
            raise ValueError(
                "defense 'sparse' needs a fitted mask: pass "
                "sparse_checkpoint=<path to the .pt written by defense/sparse.py's "
                "Sparse.save>."
            )
        from defense import Sparse

        defender = Sparse.load(checkpoint, device=str(X.device))
    return defender.protect(X, epsilon=epsilon), defender


def idct_transform(
    X: torch.Tensor,
    defender: "Any" = None,
    checkpoint: "str | Path | None" = None,
    num_subsets: int = 2,
    seed: int = 0,
) -> tuple[torch.Tensor, "Any"]:
    if defender is None:
        from defense import IdctDefense

        if checkpoint is not None:
            defender = IdctDefense.load(checkpoint, device=str(X.device))
        else:
            defender = IdctDefense(
                X.shape[1], num_subsets=num_subsets, seed=seed, device=str(X.device)
            )
    return _renorm(defender.protect(X)), defender


def cmag_transform(
    X: torch.Tensor,
    defender: "Any" = None,
    checkpoint: "str | Path | None" = None,
    epsilon: float | None = None,
) -> tuple[torch.Tensor, "Any"]:
    if defender is None:
        if checkpoint is None:
            raise ValueError(
                "defense 'cmag' needs a fitted covering: pass "
                "cmag_checkpoint=<path to the .pt written by train_algen.py "
                "--stages cmag>."
            )
        from defense import Cmag

        defender = Cmag.load(checkpoint, device=str(X.device))
    return _renorm(defender.protect(X, epsilon=epsilon)), defender


def vec2text_transform(
    X: torch.Tensor,
    defender: "Any" = None,
    checkpoint: "str | Path | None" = None,
    noise_level: float | None = None,
) -> tuple[torch.Tensor, "Any"]:
    if defender is None:
        from defense import Vec2TextDefense

        if checkpoint is not None:
            defender = Vec2TextDefense.load(checkpoint, device=str(X.device))
        else:
            defender = Vec2TextDefense(
                X.shape[1],
                **({"noise_level": noise_level} if noise_level is not None else {}),
                device=str(X.device),
            )
    return defender.protect(X, noise_level=noise_level), defender


def remote_rag_transform(
    X: torch.Tensor,
    defender: "Any" = None,
    checkpoint: "str | Path | None" = None,
    radius: float | None = None,
) -> tuple[torch.Tensor, "Any"]:
    if defender is None:
        from defense import RemoteRag

        if checkpoint is not None:
            defender = RemoteRag.load(checkpoint, device=str(X.device))
        else:
            defender = RemoteRag(
                X.shape[1],
                **({"radius": radius} if radius is not None else {}),
                device=str(X.device),
            )
    return defender.protect(X, radius=radius), defender


def keyed_rotation_transform(
    X: torch.Tensor,
    defender: "Any" = None,
    checkpoint: "str | Path | None" = None,
) -> tuple[torch.Tensor, "Any"]:
    if defender is None:
        if checkpoint is None:
            raise ValueError(
                "defense 'keyed_rotation' needs a fitted partition: pass "
                "keyed_rotation_checkpoint=<path from KeyedRotation(cfg).fit(Z).save(path)>, "
            )
        from defense.keyed_rotation import KeyedRotation

        defender = KeyedRotation.load(checkpoint, device=str(X.device))
    return defender.protect(X), defender



STATEFUL = {"shuffling", "wet", "eguard", "sparse", "idct", "cmag", "keyed_rotation"}

DEFENSES: dict[str, str] = {
    "none": "no perturbation",
    "gaussian": "additive Gaussian noise at --noise-level",
    "dp_gaussian": "(eps, delta)-DP Gaussian mechanism",
    "lapmech": "multivariate Laplace mechanism at --epsilon",
    "purmech": "Purkayastha mechanism at --epsilon",
    "shuffling": "secret dimension permutation",
    "wet": "WET circulant watermark transform",
    "masking": "overwrite leading coordinate",
    "eguard": "learned transformer projection (defense/eguard.py)",
    "sparse": "concept-aware Mahalanobis mechanism at --epsilon (defense/sparse.py)",
    "idct": "DCT-overlap spectral projection at --idct-subsets (defense/idct.py)",
    "cmag": "per-neighbourhood analytic Gaussian at --epsilon (defense/cmag.py)",
    "vec2text": "Gaussian noise at --vec2text-noise-level, unnormalised (defense/vec2text.py)",
    "remote_rag": "(n, eps)-DistanceDP at --remote-rag-radius (defense/remote_rag.py)",
    "keyed_rotation": "secret rotation per density-adaptive cell (defense/keyed_rotation.py); "
                      "in the ladder, --partition over no base defense",
}


def apply_defense(
    X: torch.Tensor,
    method: str = "none",
    *,
    noise_level: float = 0.0,
    epsilon: float = 1.0,
    delta: float = 1e-5,
    eguard_checkpoint: "str | Path | None" = None,
    sparse_checkpoint: "str | Path | None" = None,
    idct_checkpoint: "str | Path | None" = None,
    cmag_checkpoint: "str | Path | None" = None,
    vec2text_checkpoint: "str | Path | None" = None,
    vec2text_noise_level: float | None = None,
    remote_rag_checkpoint: "str | Path | None" = None,
    remote_rag_radius: float | None = None,
    idct_subsets: int = 2,
    idct_seed: int = 0,
    keyed_rotation_checkpoint: "str | Path | None" = None,
    state: dict | None = None,
) -> tuple[torch.Tensor, dict]:
    state = dict(state or {})
    method = method.lower()
    if method in ("none", ""):
        return X, state
    if method == "gaussian":
        return insert_gaussian_noise(X, noise_level), state
    if method == "dp_gaussian":
        return dp_gaussian_embeddings(X, epsilon=epsilon, delta=delta), state
    if method == "lapmech":
        return lap_mech(X, epsilon), state
    if method == "purmech":
        return pur_mech(X, epsilon), state
    if method == "masking":
        return mask_embeddings(X), state
    if method == "shuffling":
        out, perm = shuffle_embeddings(X, state.get("perm"))
        state["perm"] = perm
        return out, state
    if method == "wet":
        out, T = wet_transform(X, state.get("T"))
        state["T"] = T
        return out, state
    if method == "eguard":
        out, guard = eguard_transform(X, state.get("guard"), eguard_checkpoint)
        state["guard"] = guard
        return out, state
    if method == "sparse":
        out, defender = sparse_transform(
            X, state.get("sparse"), sparse_checkpoint, epsilon=epsilon
        )
        state["sparse"] = defender
        return out, state
    if method == "remote_rag":
        out, defender = remote_rag_transform(
            X, state.get("remote_rag"), remote_rag_checkpoint, radius=remote_rag_radius
        )
        state["remote_rag"] = defender
        return out, state
    if method == "vec2text":
        out, defender = vec2text_transform(
            X, state.get("vec2text"), vec2text_checkpoint, noise_level=vec2text_noise_level
        )
        state["vec2text"] = defender
        return out, state
    if method == "cmag":
        out, defender = cmag_transform(X, state.get("cmag"), cmag_checkpoint, epsilon=epsilon)
        state["cmag"] = defender
        return out, state
    if method == "idct":
        out, defender = idct_transform(
            X, state.get("idct"), idct_checkpoint,
            num_subsets=idct_subsets, seed=idct_seed,
        )
        state["idct"] = defender
        return out, state
    if method == "keyed_rotation":
        out, defender = keyed_rotation_transform(
            X, state.get("keyed_rotation"), keyed_rotation_checkpoint
        )
        state["keyed_rotation"] = defender
        return out, state
    raise KeyError(f"Unknown defense {method!r}. Known: {sorted(DEFENSES)}")
