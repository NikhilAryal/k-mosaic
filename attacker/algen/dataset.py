from __future__ import annotations

from typing import Any, Sequence

import torch
from torch.utils.data import Dataset

from .utils import add_punctuation_token_ids, mean_pool


class InversionDataset(Dataset):
    def __init__(
        self,
        texts: Sequence[str],
        tokenizer: Any,
        encoder: Any,
        device: torch.device | str,
        max_length: int = 32,
        lang: str = "eng",
    ) -> None:
        self.texts = list(texts)
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.device = device
        self.max_length = max_length
        self.lang = lang

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> str:
        return self.texts[idx]

    def collate_fn(self, texts: Sequence[str]) -> dict[str, Any]:
        tokens = add_punctuation_token_ids(texts, self.tokenizer, self.max_length, self.device)
        input_ids, attention_mask = tokens["input_ids"], tokens["attention_mask"]

        with torch.no_grad():
            hidden_states = self.encoder(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state
        hidden_states = mean_pool(hidden_states, attention_mask)

        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100  # CrossEntropyLoss ignore_index

        return {
            "hidden_states": hidden_states,
            "attention_mask": attention_mask,
            "input_ids": input_ids,
            "labels": labels,
            "length": attention_mask.sum(dim=1),
            "lang": self.lang,
            "text": self.tokenizer.batch_decode(input_ids, skip_special_tokens=True),
        }
