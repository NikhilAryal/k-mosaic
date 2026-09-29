from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from ..metrics import eval_texts
from .modules import (
    Discriminator,
    LinearProjection,
    MappingNetwork,
    pairwise_pivot_loss,
    sequence_cross_entropy_with_logits,
)

ARGS_FILE = "training_args_and_best_models.json"


DECODER_LR = 3e-5
DECODER_EPS = 1e-6
DECODER_WEIGHT_DECAY = 0.01
DISCRIMINATOR_LR = 1e-3
WARMUP_STEPS = 100
LABEL_SMOOTHING = 0.02
NO_DECAY = ("bias", "ln", "LayerNorm.weight")


class AdvDataset(Dataset):
    def __init__(
        self,
        texts: list[str],
        victim: np.ndarray,
        surrogate: np.ndarray,
        domains: list[int],
    ) -> None:
        self.texts = texts
        self.victim = victim.astype(np.float32)
        self.surrogate = surrogate.astype(np.float32)
        self.domains = np.asarray(domains, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, i: int):
        return self.texts[i], self.victim[i], self.surrogate[i], self.domains[i]


class EvalDataset(Dataset):
    def __init__(self, texts: list[str], victim: np.ndarray) -> None:
        self.texts = texts
        self.victim = victim.astype(np.float32)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, i: int):
        return self.texts[i], self.victim[i]


class TeiaTrainer:
    def __init__(
        self,
        decoder_name: str = "microsoft/DialoGPT-small",
        output_dir: str | Path = "attacker/outputs",
        *,
        max_length: int = 32,
        dataset: str = "nfcorpus",
        external_dataset: str = "beir/trec-covid",
        surrogate_model: str = "gte-base",
        victim_model: str = "gtr-base",
        victim_dim: int = 768,
        surrogate_dim: int = 768,
        leaked_texts: Sequence[str] | None = None,
        leaked_vectors: np.ndarray | None = None,
        leaked_surrogate: np.ndarray | None = None,
        external_texts: Sequence[str] | None = None,
        external_surrogate: np.ndarray | None = None,
        val_texts: Sequence[str] | None = None,
        val_vectors: np.ndarray | None = None,
        leaked_samples: int = 2000,
        external_samples: int = 20000,
        val_samples: int = 200,
        batch_size: int = 16,
        num_epochs: int = 24,
        eval_per_epochs: int = 2,
        mapping_lambda: float = 1.0,
        pivot_lambda: float = 1.0,
        geia: bool = False,
        defense: str = "none",
        defense_scope: str = "both",
        defense_tag: str = "",
        defense_state: dict | None = None,
        device: str | None = None,
        holdout_ids: Sequence[str] = (),
        seed: int = 42,
        build_loaders: bool = True,
    ) -> None:
        self.args: dict[str, Any] = {
            "decoder_name": decoder_name,
            "output_dir": str(output_dir),
            "max_length": max_length,
            "dataset": dataset,
            "external_dataset": external_dataset,
            "surrogate_model": surrogate_model,
            "victim_model": victim_model,
            "victim_dim": int(victim_dim),
            "surrogate_dim": int(surrogate_dim),
            "leaked_samples": leaked_samples,
            "external_samples": external_samples,
            "val_samples": val_samples,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "eval_per_epochs": eval_per_epochs,
            "mapping_lambda": mapping_lambda,
            "pivot_lambda": pivot_lambda,
            "geia": bool(geia),
            "defense": defense,
            "defense_scope": defense_scope,
            "defense_tag": defense_tag,
            "seed": seed,
            "holdout_ids": sorted(holdout_ids),
        }

        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.geia = bool(geia)

        self.defense_state: dict = dict(defense_state or {})

        self.tokenizer = AutoTokenizer.from_pretrained(decoder_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(decoder_name).to(self.device)

        hidden = self.model.config.hidden_size
        self.projection = LinearProjection(victim_dim, hidden).to(self.device)
        self.mapping = MappingNetwork(surrogate_dim, victim_dim).to(self.device)
        self.discriminator = Discriminator(victim_dim).to(self.device)
        self.bce = nn.BCELoss()
        self.mse = nn.MSELoss()

        self.optimizer = self._build_optimizer()
        self.d_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=DISCRIMINATOR_LR
        )

        self.run_dir = self.run_dir_for(**self.args)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.best_models: list[tuple[float, dict, str]] = []
        self.best_score = float("inf")

        self.train_loader = self.val_loader = None
        if build_loaders:
            self._build_loaders(
                leaked_texts or [], leaked_vectors, leaked_surrogate,
                external_texts or [], external_surrogate,
                val_texts or [], val_vectors,
            )

    def _build_optimizer(self):
        from torch.optim import AdamW

        named = list(self.model.named_parameters())
        groups = [
            {
                "params": [p for n, p in named if not any(nd in n for nd in NO_DECAY)],
                "weight_decay": DECODER_WEIGHT_DECAY,
            },
            {
                "params": [p for n, p in named if any(nd in n for nd in NO_DECAY)],
                "weight_decay": 0.0,
            },
        ]
        opt = AdamW(groups, lr=DECODER_LR, eps=DECODER_EPS)
        opt.add_param_group({"params": list(self.projection.parameters())})
        opt.add_param_group({"params": list(self.mapping.parameters())})
        return opt

    @staticmethod
    def run_dir_for(
        *,
        decoder_name: str,
        output_dir: str | Path,
        dataset: str,
        external_dataset: str,
        surrogate_model: str,
        max_length: int,
        leaked_samples: int,
        external_samples: int,
        batch_size: int,
        num_epochs: int,
        geia: bool = False,
        defense_tag: str = "",
        **_ignored: Any,
    ) -> Path:
        arm = "geia" if geia else f"teia_{surrogate_model}"
        ext = "" if geia else f"_ext{str(external_dataset).replace('/', '_')}{external_samples}"
        tag = f"_{defense_tag}" if defense_tag else ""
        return Path(output_dir) / "teia" / decoder_name.replace("/", "_") / (
            f"{str(dataset).replace('/', '_')}_maxlength{max_length}"
            f"_{arm}_leaked{leaked_samples}{ext}"
            f"_batch_size{batch_size}_epochs{num_epochs}{tag}"
        )

    @staticmethod
    def is_complete(run_dir: str | Path) -> bool:
        args_path = Path(run_dir) / ARGS_FILE
        if not args_path.exists():
            return False
        try:
            with open(args_path) as f:
                best = json.load(f).get("best_models") or []
        except (OSError, json.JSONDecodeError):
            return False
        return bool(best) and Path(best[0][-1]).exists()

    def _build_loaders(
        self,
        leaked_texts, leaked_vectors, leaked_surrogate,
        external_texts, external_surrogate,
        val_texts, val_vectors,
    ) -> None:
        leaked_texts = list(leaked_texts)
        texts = list(leaked_texts)
        victim = np.asarray(leaked_vectors, dtype=np.float32)
        surrogate = np.asarray(leaked_surrogate, dtype=np.float32)
        domains = [0] * len(leaked_texts)

        if self.geia:
            # The direct-attack baseline: leaked rows only, no sampler reweighting.
            dataset = AdvDataset(texts, victim, surrogate, domains)
            self.train_loader = DataLoader(
                dataset, batch_size=self.batch_size, shuffle=True
            )
        else:
            external_texts = list(external_texts)
            ext_surrogate = np.asarray(external_surrogate, dtype=np.float32)
            texts = texts + external_texts
            victim = np.concatenate(
                [victim, np.zeros((len(external_texts), victim.shape[1]), np.float32)]
            )
            surrogate = np.concatenate([surrogate, ext_surrogate])
            domains = domains + [1] * len(external_texts)

            weights = [len(external_texts)] * len(leaked_texts) + [
                len(leaked_texts)
            ] * len(external_texts)
            dataset = AdvDataset(texts, victim, surrogate, domains)
            self.train_loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                sampler=WeightedRandomSampler(weights, len(dataset), replacement=True),
            )

        self.val_loader = DataLoader(
            EvalDataset(list(val_texts), np.asarray(val_vectors, dtype=np.float32)),
            batch_size=self.batch_size,
            shuffle=False,
        )
        n_ext = 0 if self.geia else len(external_texts)
        print(
            f"[teia] loaders: leaked={len(leaked_texts)} external={n_ext} "
            f"val={len(val_texts)} batch={self.batch_size}"
        )

    def _lm_loss(self, embeddings: torch.Tensor, texts: list[str]):
        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        input_ids = enc["input_ids"].to(self.device)

        token_emb = self.model.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([embeddings.unsqueeze(1), token_emb], dim=1)

        logits = self.model(inputs_embeds=inputs_embeds, return_dict=True).logits
        logits = logits[:, :-1].contiguous()
        target = input_ids.contiguous()

        mask = torch.ones_like(target).float()
        loss = sequence_cross_entropy_with_logits(
            logits, target, mask, label_smoothing=LABEL_SMOOTHING, reduce="batch"
        )
        return loss, float(np.exp(min(loss.item(), 20.0)))

    def _pivot_loss(self, real: torch.Tensor, mapped: torch.Tensor) -> torch.Tensor:
        return self.mse(real, mapped) + pairwise_pivot_loss(real, mapped)


    def train(self) -> list[tuple[float, dict, str]]:
        from transformers import get_linear_schedule_with_warmup

        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=WARMUP_STEPS,
            num_training_steps=max(1, len(self.train_loader) * self.num_epochs),
        )
        print(
            f"TEIA training {'GEIA baseline' if self.geia else 'TEIA'} for "
            f"{self.num_epochs} epochs -> {self.run_dir}"
        )

        for epoch in range(self.num_epochs):
            if not self.geia:
                self._discriminator_epoch()
            stats = self._generator_epoch(scheduler)

            if (epoch + 1) % self.args["eval_per_epochs"] == 0 or epoch + 1 == self.num_epochs:
                metrics = self.validate()
                print(
                    f"[epoch {epoch + 1}/{self.num_epochs}] "
                    + " ".join(f"{k}={v:.4f}" for k, v in stats.items())
                    + " | val "
                    + " ".join(
                        f"{k}={v:.4f}" for k, v in metrics.items() if isinstance(v, float)
                    )
                )
                self.save_best_model(metrics, epoch + 1, stats)
        return self.best_models

    def _discriminator_epoch(self) -> None:
        self.mapping.requires_grad_(False)
        self.discriminator.requires_grad_(True)
        for texts, embs, s_embs, domains in self.train_loader:
            real = embs[domains == 0].to(self.device)
            mapped_src = s_embs[domains == 1].to(self.device)
            if not len(real) or not len(mapped_src):
                continue  # a batch from one domain only: nothing to discriminate
            with torch.no_grad():
                mapped = self.mapping(mapped_src)
            pred = torch.cat([self.discriminator(real), self.discriminator(mapped)])
            labels = torch.cat(
                [torch.zeros(len(real), 1), torch.ones(len(mapped), 1)]
            ).to(self.device)
            loss = self.bce(pred, labels)
            self.d_optimizer.zero_grad()
            loss.backward()
            self.d_optimizer.step()

    def _generator_epoch(self, scheduler) -> dict[str, float]:
        self.mapping.requires_grad_(True)
        self.discriminator.requires_grad_(False)
        self.model.train()

        totals = {"lm": 0.0, "pivot": 0.0, "adv": 0.0, "ppl": 0.0}
        n = 0
        for texts, embs, s_embs, domains in self.train_loader:
            leaked = domains == 0
            external = domains == 1
            text_l = [t for t, m in zip(texts, leaked.tolist()) if m]
            text_e = [t for t, m in zip(texts, external.tolist()) if m]
            real = embs[leaked].to(self.device)

            if self.geia:
                if not len(real):
                    continue
                lm_loss, ppl = self._lm_loss(self.projection(real), text_l)
                loss = lm_loss
                pivot = adv = torch.zeros((), device=self.device)
            else:
                if not len(real) or not external.any():
                    continue
                mapped_leaked = self.mapping(s_embs[leaked].to(self.device))
                mapped_ext = self.mapping(s_embs[external].to(self.device))

                pivot = self._pivot_loss(real, mapped_leaked)
                adv = self.bce(
                    self.discriminator(mapped_ext),
                    torch.zeros(len(mapped_ext), 1, device=self.device),
                )
                lm_loss, ppl = self._lm_loss(
                    self.projection(torch.cat([real, mapped_ext])), text_l + text_e
                )
                loss = (
                    lm_loss
                    + self.args["mapping_lambda"] * adv
                    + self.args["pivot_lambda"] * pivot
                )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            scheduler.step()

            totals["lm"] += lm_loss.item()
            totals["pivot"] += pivot.detach().item()
            totals["adv"] += adv.detach().item()
            totals["ppl"] += ppl
            n += 1

        return {k: v / max(n, 1) for k, v in totals.items()}


    @torch.no_grad()
    def generate(self, vectors: torch.Tensor, batch_size: int = 32) -> list[str]:
        self.model.eval()
        out: list[str] = []
        for start in range(0, len(vectors), batch_size):
            chunk = vectors[start : start + batch_size].to(self.device, torch.float32)
            prefix = self.projection(chunk).unsqueeze(1)
            generated = self.model.generate(
                inputs_embeds=prefix,
                attention_mask=torch.ones(
                    prefix.shape[:2], dtype=torch.long, device=self.device
                ),
                max_new_tokens=self.max_length,
                do_sample=True,
                temperature=0.9,
                top_p=0.9,
                top_k=0,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            out.extend(
                t.strip()
                for t in self.tokenizer.batch_decode(generated, skip_special_tokens=True)
            )
        return out

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        self.model.eval()
        preds, refs, ppls = [], [], []
        for texts, embs in self.val_loader:
            embs = embs.to(self.device)
            _, ppl = self._lm_loss(self.projection(embs), list(texts))
            ppls.append(ppl)
            preds.extend(self.generate(embs))
            refs.extend(list(texts))
        metrics = eval_texts(preds, refs)
        metrics["perplexity"] = float(np.mean(ppls)) if ppls else float("nan")
        return metrics


    def save_best_model(self, metrics: dict, epoch: int, train_stats: dict) -> None:
        score = -float(metrics.get("rougeL", 0.0))
        if score >= self.best_score:
            print(f"  rougeL {-score:.4f} <= best {-self.best_score:.4f} — not saving")
            return

        self.best_score = score
        ckpt = self.run_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "projection_state_dict": self.projection.state_dict(),
                "mapping_state_dict": self.mapping.state_dict(),
                "discriminator_state_dict": self.discriminator.state_dict(),

                "defense_state": {
                    k: v for k, v in self.defense_state.items()
                    if isinstance(v, torch.Tensor)
                },
                "val_results": metrics,
                "train_stats": train_stats,
            },
            ckpt,
        )
        self.best_models.append((score, metrics, str(ckpt)))
        self.best_models.sort(key=lambda x: x[0])
        while len(self.best_models) > 2:
            _, _, stale = self.best_models.pop()
            Path(stale).unlink(missing_ok=True)

        with open(self.run_dir / ARGS_FILE, "w") as f:
            json.dump(
                {"training_args": self.args, "best_models": self.best_models}, f, indent=4
            )
        print(f" saved best model (rougeL={-score:.4f}) -> {ckpt}")

    def load_best_model(self) -> str:
        if not self.best_models:
            raise ValueError(f"no checkpoints recorded in {self.run_dir / ARGS_FILE}")
        self.best_score, _, path = self.best_models[0]
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model_state_dict"])
        self.projection.load_state_dict(state["projection_state_dict"])
        self.mapping.load_state_dict(state["mapping_state_dict"])
        self.discriminator.load_state_dict(state["discriminator_state_dict"])
        self.defense_state.update(
            {k: v.to(self.device) for k, v in (state.get("defense_state") or {}).items()}
        )
        print(f"TEIA loaded {path} (val rougeL={-self.best_score:.4f})")
        return path

    @classmethod
    def from_checkpoint(
        cls, checkpoint_dir: str | Path, device: str | None = None
    ) -> "TeiaTrainer":
        checkpoint_dir = Path(checkpoint_dir)
        args_path = checkpoint_dir / ARGS_FILE
        if not args_path.exists():
            raise FileNotFoundError(
                f"{args_path} not found — point --checkpoint at a run directory "
                "produced by `python train_teia.py --stages train`."
            )
        with open(args_path) as f:
            data = json.load(f)
        args = dict(data["training_args"])

        trainer = cls(
            **{
                k: args[k]
                for k in (
                    "decoder_name", "output_dir", "max_length", "dataset",
                    "external_dataset", "surrogate_model", "victim_model",
                    "victim_dim", "surrogate_dim", "leaked_samples",
                    "external_samples", "val_samples", "batch_size", "num_epochs",
                    "eval_per_epochs", "mapping_lambda", "pivot_lambda", "geia",
                    "defense", "defense_scope", "defense_tag", "seed",
                )
                if k in args
            },
            holdout_ids=args.get("holdout_ids", []),
            device=device,
            build_loaders=False,
        )
        trainer.best_models = [tuple(b) for b in data["best_models"]]
        trainer.load_best_model()
        return trainer
