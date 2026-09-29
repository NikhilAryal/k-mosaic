"""The LLM prior: where Zero2Text's pairs come from when nothing is leaked.

Every other attack in this repo obtains its ``(text, vector)`` pairs from a corpus
the attacker is assumed to own — ALGEN's ``align_samples`` slice, TEIA's leaked
``D_L``. Zero2Text assumes neither. Its pairs are **manufactured**: an LLM writes
text, the attacker pushes that text through its query access to the victim service,
and the pair that comes back is as good as a leaked one for fitting a map.

That is the whole reason the method matters here. ``P`` (leaked pairs) is the axis
this project characterises defenses on, and a training-free attacker appears to set
``P = 0``. It does not. It converts pairs into *queries* — see
:class:`~attacker.zero2text.attack.Zero2TextAttacker` for the accounting.

Two operations, matching the paper's two phases:

``seed``
    Unconditioned generation from generic prompts. No target information enters,
    which is what makes this the cross-domain setting: the pool is whatever the LLM
    writes, not whatever the victim corpus contains.
``vary``
    Paraphrase around the current best reconstruction per target. This is the
    "recursive" half of recursive online alignment — the pool migrates toward the
    targets, so the ridge map is fitted where it is about to be evaluated.

Research use only.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

import torch

#: Generic openers for the unconditioned seed round.
#:
#: Deliberately contentless. A prompt naming the victim corpus's domain ("write a
#: medical question") would smuggle in exactly the in-domain knowledge the method
#: claims not to need, and the resulting numbers would describe a different attack.
#: ZSInvert (arXiv:2504.00147 §4) uses a single ``"tell me a story"``; a small cycle
#: buys pool diversity at no extra assumption.
SEED_PROMPTS = (
    "Write one short sentence.",
    "Write a short question someone might ask.",
    "Write one sentence of factual information.",
    "Write a short sentence about anything at all.",
    "State one fact in a single sentence.",
    "Ask one short question.",
)

#: Paraphrase instruction for the refinement rounds (ZSInvert Stage 2's prompt).
VARY_PROMPT = "Write a sentence similar to: {text}"

_WS = re.compile(r"\s+")


def _clean(text: str) -> str:
    """One line, no quotes, no list markers — the LLM's framing is not content."""
    text = _WS.sub(" ", text).strip()
    text = text.split("\n")[0].strip()
    text = text.strip('"').strip("'").strip()
    text = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", text)
    return text.strip()


class LLMProposer:
    """A causal LM that writes candidate texts, with no access to the victim corpus."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
        *,
        device: str | torch.device | None = None,
        max_new_tokens: int = 32,
        temperature: float = 1.0,
        top_p: float = 0.95,
        batch_size: int = 64,
        seed: int = 42,
        dtype: str = "bfloat16",
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.batch_size = batch_size
        self.seed = seed

        self.device = torch.device(
            device if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Decoder-only generation needs left padding, or the batch's shorter prompts
        # are continued from pad tokens and come back as gibberish.
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=getattr(torch, dtype, torch.float32)
        ).to(self.device).eval()
        self._calls = 0
        print(f"[z2t/llm] {model_id} on {self.device} "
              f"(max_new_tokens={max_new_tokens}, T={temperature}, top_p={top_p})")

    # ------------------------------------------------------------------ #

    def _render(self, instruction: str) -> str:
        """Apply the chat template when the checkpoint has one."""
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": instruction}],
                tokenize=False, add_generation_prompt=True,
            )
        return instruction + "\n"

    @torch.no_grad()
    def _generate(self, instructions: Sequence[str]) -> list[str]:
        out: list[str] = []
        for start in range(0, len(instructions), self.batch_size):
            chunk = [self._render(p) for p in instructions[start : start + self.batch_size]]
            enc = self.tokenizer(chunk, return_tensors="pt", padding=True).to(self.device)
            gen = torch.Generator(device="cpu").manual_seed(self.seed + self._calls)
            self._calls += 1
            ids = self.model.generate(
                **enc,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                # torch.Generator keeps the pool reproducible across runs; without it
                # a cached arm and a fresh one are different attacks under one name.
                **({"generator": gen} if _accepts_generator(self.model) else {}),
            )
            new = ids[:, enc["input_ids"].shape[1] :]
            out += [_clean(t) for t in self.tokenizer.batch_decode(new, skip_special_tokens=True)]
        return out

    # ------------------------------------------------------------------ #

    def seed_pool(self, n: int) -> list[str]:
        """``n`` unconditioned candidates. No target information reaches this."""
        prompts = [SEED_PROMPTS[i % len(SEED_PROMPTS)] for i in range(n)]
        texts = [t for t in self._generate(prompts) if t]
        print(f"[z2t/llm] seeded {len(texts)}/{n} candidate(s)")
        return texts

    def vary(self, texts: Sequence[str], n_per: int = 1) -> list[str]:
        """``n_per`` paraphrases of each input — the recursive half of the loop."""
        prompts = [VARY_PROMPT.format(text=t) for t in texts for _ in range(n_per)]
        out = [t for t in self._generate(prompts) if t]
        print(f"[z2t/llm] varied {len(texts)} best-so-far -> {len(out)} candidate(s)")
        return out

    def free(self) -> None:
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _accepts_generator(model: Any) -> bool:
    """transformers exposes ``generator=`` on sampling only in some versions."""
    import inspect

    try:
        return "generator" in inspect.signature(model.generate).parameters
    except (TypeError, ValueError):
        return False
