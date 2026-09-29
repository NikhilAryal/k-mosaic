from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import get_device, load_encoder_decoder_and_tokenizer


class InversionGenerator(nn.Module):
    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        max_length: int = 32,
        device: torch.device | str | None = None,
        num_beams: int = 3,
        repetition_penalty: float = 2.0,
        length_penalty: float = 2.0,
    ) -> None:
        super().__init__()
        self.device = get_device(device) if not isinstance(device, torch.device) else device
        self.model_name = model_name
        self.encoder_decoder, self.tokenizer = load_encoder_decoder_and_tokenizer(
            model_name, self.device
        )

        self.embedder_dim = self.encoder_decoder.config.hidden_size
        self.num_repeat_tokens = max_length
        self.max_length = max_length
        # transformers >= 5 rejects generation knobs left on `config`; they live on
        self.encoder_decoder.generation_config.max_length = max_length
        self.num_beams = num_beams
        self.repetition_penalty = repetition_penalty
        self.length_penalty = length_penalty

        dropout_rate = getattr(self.encoder_decoder.config, "dropout_rate", 0.1)
        self.embedding_transform = nn.Sequential(
            nn.Linear(self.embedder_dim, self.embedder_dim),
            nn.Dropout(dropout_rate),
            nn.GELU(),
            nn.Linear(self.embedder_dim, self.embedder_dim * self.num_repeat_tokens),
        ).to(self.device)

    def get_embeddings(
        self, embeddings: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = embeddings.to(self.device)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        repeated = self.embedding_transform(embeddings)
        return repeated.reshape((*repeated.shape[:-1], self.num_repeat_tokens, -1)), attention_mask

    def _start_ids(self, batch_size: int) -> torch.Tensor:
        bos = self.tokenizer.bos_token_id
        start = bos if bos is not None else self.tokenizer.eos_token_id
        return torch.full((batch_size, 1), start, dtype=torch.long, device=self.device)


    def forward(self, inputs: Dict[str, torch.Tensor]) -> Any:
        input_embeds, attention_mask = self.get_embeddings(
            inputs["hidden_states"], inputs["attention_mask"]
        )
        batch_size = input_embeds.size(0)
        assert batch_size == inputs["labels"].size(0), (
            f"Batch size mismatch: inputs_embeds={batch_size}, "
            f"labels={inputs['labels'].size(0)}"
        )
        return self.encoder_decoder(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            labels=inputs["labels"],
        )

    @torch.no_grad()
    def generate(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        input_embeds, attention_mask = self.get_embeddings(
            inputs["hidden_states"], inputs["attention_mask"]
        )
        return self.encoder_decoder.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            decoder_input_ids=self._start_ids(input_embeds.size(0)),
            max_length=self.max_length,
            num_beams=self.num_beams,
            repetition_penalty=self.repetition_penalty,
            length_penalty=self.length_penalty,
            early_stopping=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

    @property
    def encoder(self) -> Any:
        return self.encoder_decoder.encoder
