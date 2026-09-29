from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import InversionDataset
from .generator import InversionGenerator
from ..metrics import eval_texts

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ARGS_FILE = "training_args_and_best_models.json"


class GeneratorTrainer:
    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        output_dir: str | Path = "attacker/outputs",
        *,
        max_length: int = 32,
        dataset: str = "nfcorpus",
        train_texts: Sequence[str] | None = None,
        val_texts: Sequence[str] | None = None,
        train_samples: int = 3000,
        val_samples: int = 200,
        batch_size: int = 64,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        num_epochs: int = 30,
        wandb_run_name: str | None = None,
        device: str | None = None,
        build_loaders: bool = True,
        holdout_ids: Sequence[str] = (),
        align_reserve: int = 200,
        seed: int = 42,
    ) -> None:
        self.args = {
            "model_name": model_name,
            "output_dir": str(output_dir),
            "max_length": max_length,
            "dataset": dataset,
            "train_samples": train_samples,
            "val_samples": val_samples,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "num_epochs": num_epochs,
            "align_reserve": align_reserve,
            "seed": seed,
            "holdout_ids": sorted(holdout_ids),
        }

        self.model = InversionGenerator(model_name, max_length=max_length, device=device)
        self.device = self.model.device
        self.tokenizer = self.model.tokenizer
        self.encoder = self.model.encoder
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        self.run_dir = self.run_dir_for(**self.args)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.best_models: list[tuple[float, dict, str]] = []
        self.best_val_loss = float("inf")
        self.start_epoch = 0
        self._wandb = None

        if build_loaders:
            self._build_loaders(train_texts or [], val_texts or [])
        if wandb_run_name:
            self._init_wandb(wandb_run_name)


    @staticmethod
    def run_dir_for(
        *,
        model_name: str,
        output_dir: str | Path,
        dataset: str,
        max_length: int,
        train_samples: int,
        batch_size: int,
        learning_rate: float,
        weight_decay: float,
        num_epochs: int,
        **_ignored: Any,
    ) -> Path:

        return Path(output_dir) / model_name.replace("/", "_") / (
            f"{str(dataset).replace('/', '_')}_maxlength{max_length}"
            f"_train{train_samples}_batch_size{batch_size}"
            f"_lr{learning_rate}_wd{weight_decay}_epochs{num_epochs}"
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

    def _build_loaders(self, train_texts: Sequence[str], val_texts: Sequence[str]) -> None:
        train_texts = list(train_texts)[: self.args["train_samples"]]
        val_texts = list(val_texts)[: self.args["val_samples"]]
        self.train_dataset = InversionDataset(
            train_texts, self.tokenizer, self.encoder, self.device, self.max_length
        )
        self.val_dataset = InversionDataset(
            val_texts, self.tokenizer, self.encoder, self.device, self.max_length
        )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.train_dataset.collate_fn,
            num_workers=0,  # collate_fn runs the live (still-training) encoder
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.val_dataset.collate_fn,
            num_workers=0,
        )

    def _init_wandb(self, run_name: str) -> None:
        try:
            import wandb
        except ImportError:
            print("[wandb] not installed — logging to stdout only")
            return
        self._wandb = wandb
        wandb.init(
            project=self.args["model_name"].replace("/", "_") + "_inversion",
            name=run_name,
            config=self.args,
        )

    def train(self) -> list[tuple[float, dict, str]]:
        try:
            for epoch in range(self.start_epoch, self.num_epochs):
                self.model.train()
                epoch_loss = 0.0
                bar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.num_epochs}")
                for batch in bar:
                    outputs = self.model(
                        {
                            "hidden_states": batch["hidden_states"].to(self.device),
                            "attention_mask": batch["attention_mask"].to(self.device),
                            "labels": batch["labels"].to(self.device),
                        }
                    )
                    loss = outputs.loss
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    epoch_loss += loss.item()
                    bar.set_postfix({"loss": loss.item()})

                avg_loss = epoch_loss / max(len(self.train_loader), 1)
                val_loss, gen_results = self.validate()
                print(
                    f"Epoch {epoch + 1}/{self.num_epochs} "
                    f"train_loss={avg_loss:.4f} val_loss={val_loss:.4f} {gen_results}"
                )
                if self._wandb is not None and self._wandb.run:
                    self._wandb.log(
                        {"epoch": epoch + 1, "train_loss": avg_loss, "val_loss": val_loss, **gen_results}
                    )
                self.save_best_model(val_loss, gen_results, epoch + 1)
        finally:
            if self._wandb is not None and self._wandb.run:
                self._wandb.finish()
        return self.best_models

    @torch.no_grad()
    def validate(self) -> tuple[float, dict[str, float]]:
        self.model.eval()
        val_loss = 0.0
        predictions: list[str] = []
        references: list[str] = []

        for batch in tqdm(self.val_loader, desc="Validation", leave=False):
            inputs = {
                "hidden_states": batch["hidden_states"].to(self.device),
                "attention_mask": batch["attention_mask"].to(self.device),
                "labels": batch["labels"].to(self.device),
            }
            val_loss += self.model(inputs).loss.item()
            generated = self.model.generate(inputs)
            decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
            predictions += [t.strip() for t in decoded]
            references += batch["text"]

        if predictions:
            print(f"  decoded: {predictions[:2]}")
            print(f"  true:    {references[:2]}")
        return val_loss, eval_texts(predictions, references) if predictions else {}


    def save_best_model(self, val_loss: float, val_results: dict, epoch: int) -> None:
        if val_loss >= self.best_val_loss:
            print(f"  val_loss {val_loss:.4f} >= best {self.best_val_loss:.4f} — not saving")
            return

        self.best_val_loss = val_loss
        ckpt = self.run_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_loss": val_loss,
                "val_results": val_results,
            },
            ckpt,
        )

        self.best_models.append((val_loss, val_results, str(ckpt)))
        self.best_models.sort(key=lambda x: x[0])
        while len(self.best_models) > 2:  # keep the top 2, as upstream does
            _, _, stale = self.best_models.pop()
            Path(stale).unlink(missing_ok=True)

        with open(self.run_dir / ARGS_FILE, "w") as f:
            json.dump({"training_args": self.args, "best_models": self.best_models}, f, indent=4)
        print(f"  saved best model (val_loss={val_loss:.4f}) -> {ckpt}")

    def load_best_model(self) -> str:
        if not self.best_models:
            raise ValueError(f"no checkpoints recorded in {self.run_dir / ARGS_FILE}")
        self.best_val_loss, _, best_path = self.best_models[0]
        state = torch.load(best_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model_state_dict"])
        if "optimizer_state_dict" in state:
            try:
                self.optimizer.load_state_dict(state["optimizer_state_dict"])
            except ValueError:
                pass  # optimizer state is irrelevant for inference
        self.start_epoch = state.get("epoch", 0)
        print(f"generator loaded {best_path} (val_loss={self.best_val_loss:.4f})")
        return best_path


    def init_from(self, checkpoint_dir: str | Path) -> str:
        checkpoint_dir = Path(checkpoint_dir)
        with open(checkpoint_dir / ARGS_FILE) as f:
            prior = json.load(f)
        for key in ("model_name", "max_length"):
            if prior["training_args"][key] != self.args[key]:
                raise ValueError(
                    f"cannot warm-start: {key} differs "
                    f"({prior['training_args'][key]!r} vs {self.args[key]!r})"
                )
        best_path = prior["best_models"][0][-1]
        state = torch.load(best_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model_state_dict"])
        print(f"generator warm-started from {best_path}")
        return best_path

    @classmethod
    def from_checkpoint(
        cls, checkpoint_dir: str | Path, device: str | None = None
    ) -> "GeneratorTrainer":
        checkpoint_dir = Path(checkpoint_dir)
        args_path = checkpoint_dir / ARGS_FILE
        if not args_path.exists():
            raise FileNotFoundError(
                f"{args_path} not found — point --checkpoint at a run directory "
                "produced by `python -m attacker train`."
            )
        with open(args_path) as f:
            data = json.load(f)
        args = data["training_args"]

        trainer = cls(
            model_name=args["model_name"],
            output_dir=args["output_dir"],
            max_length=args["max_length"],
            dataset=args.get("dataset", "nfcorpus"),
            train_samples=args["train_samples"],
            val_samples=args["val_samples"],
            batch_size=args["batch_size"],
            learning_rate=args["learning_rate"],
            weight_decay=args["weight_decay"],
            num_epochs=args["num_epochs"],
            device=device,
            build_loaders=False,
            holdout_ids=args.get("holdout_ids", ()),
            align_reserve=args.get("align_reserve", 200),
            seed=args.get("seed", 42),
        )
        trainer.run_dir = checkpoint_dir
        trainer.best_models = [tuple(m) for m in data["best_models"]]  # type: ignore[misc]
        trainer.load_best_model()
        trainer.model.eval()
        return trainer
