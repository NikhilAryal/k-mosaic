from __future__ import annotations

import random
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer

from ..utils import get_device, set_seed  




def mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    B, S, D = hidden_states.shape
    unmasked = hidden_states * attention_mask[..., None]
    pooled = unmasked.sum(dim=1) / attention_mask.sum(dim=1)[:, None]
    assert pooled.shape == (B, D)
    return pooled


def check_normalization(embeddings: torch.Tensor, name: str) -> dict[str, float]:
    norms = torch.norm(embeddings, p=2, dim=1)
    stats = {
        "mean_norm": float(norms.mean()),
        "std_norm": float(norms.std()) if norms.numel() > 1 else 0.0,
    }
    print(f"norm {name}: mean={stats['mean_norm']:.4f} std={stats['std_norm']:.4f}")
    return stats


def fill_in_pad_eos_token(tokenizer: Any) -> int:
    added = 0
    if tokenizer.pad_token is None:
        added += tokenizer.add_special_tokens({"pad_token": "<pad>"})
    if tokenizer.eos_token is None:
        added += tokenizer.add_special_tokens({"eos_token": "</s>"})
    return added


def load_encoder_decoder_and_tokenizer(
    model_name: str, device: torch.device | str
) -> tuple[Any, Any]:
    encoder_decoder = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if fill_in_pad_eos_token(tokenizer):
        encoder_decoder.resize_token_embeddings(len(tokenizer))
    return encoder_decoder.to(device), tokenizer


def load_source_encoder_and_tokenizer(
    model_name: str, device: torch.device | str
) -> tuple[Any, Any]:
    encoder = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if fill_in_pad_eos_token(tokenizer):
        encoder.resize_token_embeddings(len(tokenizer))
    return encoder.to(device), tokenizer


def add_punctuation_token_ids(
    sentence: Sequence[str] | str,
    tokenizer: Any,
    max_length: int,
    device: torch.device | str,
    punctuations: Sequence[str] = (".", "?", "!"),
) -> dict[str, torch.Tensor]:
    punct_token_ids = tokenizer.convert_tokens_to_ids(list(punctuations))
    approved = punct_token_ids + [tokenizer.eos_token_id, tokenizer.pad_token_id]
    period_token_id = punct_token_ids[0]

    tokens = tokenizer(
        list(sentence) if not isinstance(sentence, str) else sentence,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = tokens["input_ids"]
    mask = ~torch.isin(input_ids[:, -2], torch.tensor(approved))
    input_ids[mask, -2] = period_token_id
    return {
        "input_ids": input_ids.to(device),
        "attention_mask": tokens["attention_mask"].to(device),
    }


def get_Y_embeddings_from_tokens(
    tokens: dict[str, torch.Tensor], target_encoder: Any, normalization: bool = True
) -> torch.Tensor:
    with torch.no_grad():
        out = target_encoder(**tokens).last_hidden_state
    pooled = mean_pool(out, tokens["attention_mask"])
    return F.normalize(pooled, p=2, dim=1) if normalization else pooled
