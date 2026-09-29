from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class LinearProjection(nn.Module):

    def __init__(self, in_num: int, out_num: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_num, out_num)

    def forward(self, embs: torch.Tensor) -> torch.Tensor:
        return torch.clamp(self.fc1(embs), min=-1e9, max=1e9)


class MappingNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, (input_dim + output_dim) // 2),
            nn.ReLU(),
            nn.Linear((input_dim + output_dim) // 2, output_dim),
        )

    def forward(self, embs: torch.Tensor) -> torch.Tensor:
        return self.net(embs.float())


class Discriminator(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, embs: torch.Tensor) -> torch.Tensor:
        return self.net(embs)


def pairwise_pivot_loss(emb1: torch.Tensor, emb2: torch.Tensor) -> torch.Tensor:
    q1 = F.normalize(emb1, dim=-1)
    q2 = F.normalize(emb2, dim=-1)
    return F.mse_loss(q1 @ q1.T, q2 @ q2.T)


def sequence_cross_entropy_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    label_smoothing: float = -1.0,
    reduce: str | None = None,
) -> torch.Tensor:
    logits_flat = logits.view(-1, logits.size(-1))
    log_probs_flat = F.log_softmax(logits_flat, dim=-1)
    targets_flat = targets.reshape(-1, 1).long()

    if label_smoothing > 0.0:
        num_classes = logits.size(-1)
        smoothing_value = label_smoothing / float(num_classes)
        one_hot = torch.zeros_like(log_probs_flat).scatter_(
            -1, targets_flat, 1.0 - label_smoothing
        )
        smoothed = one_hot + smoothing_value
        nll_flat = -(log_probs_flat * smoothed).sum(-1, keepdim=True)
    else:
        nll_flat = -torch.gather(log_probs_flat, dim=1, index=targets_flat)

    nll = nll_flat.view(-1, logits.shape[1])
    loss = nll * mask
    if reduce:
        loss = loss.sum(1) / (mask.sum(1) + 1e-13)
        if reduce == "batch":
            loss = loss.mean()
    return loss


__all__ = [
    "Discriminator",
    "LinearProjection",
    "MappingNetwork",
    "pairwise_pivot_loss",
    "sequence_cross_entropy_with_logits",
]
