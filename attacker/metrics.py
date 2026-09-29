from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Sequence

import numpy as np
import torch
from rouge_score import rouge_scorer
from torch.nn import MSELoss

_ROUGE = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
_MSE = MSELoss()
_COS = torch.nn.CosineSimilarity(dim=1)


def get_rouge_scores(predictions: Sequence[str], references: Sequence[str]) -> dict[str, list[float]]:
    metrics: dict[str, list[float]] = defaultdict(list)
    for pred, target in zip(predictions, references):
        for metric, score in _ROUGE.score(target, pred).items():
            metrics[f"{metric}_f"].append(score.fmeasure)
    return metrics


_TOKEN = re.compile(r"\w+")


def token_prf(predictions: Sequence[str], references: Sequence[str]) -> dict[str, float]:
    if not len(predictions):
        return {"token_precision": 0.0, "token_recall": 0.0, "token_f1": 0.0}

    totals = [0.0, 0.0, 0.0]
    for pred, ref in zip(predictions, references):
        p_counts = Counter(_TOKEN.findall(str(pred).lower()))
        r_counts = Counter(_TOKEN.findall(str(ref).lower()))
        n_pred, n_ref = sum(p_counts.values()), sum(r_counts.values())
        overlap = sum((p_counts & r_counts).values())
        precision = overlap / n_pred if n_pred else 0.0
        recall = overlap / n_ref if n_ref else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        totals[0] += precision
        totals[1] += recall
        totals[2] += f1

    n = len(predictions)
    return {
        "token_precision": round(totals[0] / n, 4),
        "token_recall": round(totals[1] / n, 4),
        "token_f1": round(totals[2] / n, 4),
    }


def eval_texts(predictions: Sequence[str], references: Sequence[str]) -> dict[str, float]:
    import sacrebleu

    predictions = [p if p else " " for p in predictions]
    bleu = sacrebleu.corpus_bleu(list(predictions), [list(references)])
    rouge = {k: float(np.mean(v)) for k, v in get_rouge_scores(predictions, references).items()}
    exact = np.array(predictions) == np.array(references)
    return {
        **token_prf(predictions, references),
        "bleu": round(bleu.score, 2),
        "bleu1": round(bleu.precisions[0], 2),
        "bleu2": round(bleu.precisions[1], 2),
        "bleu3": round(bleu.precisions[2], 2),
        "bleu4": round(bleu.precisions[3], 2),
        "rougeL": round(rouge["rougeL_f"], 4),
        "rouge1": round(rouge["rouge1_f"], 4),
        "rouge2": round(rouge["rouge2_f"], 4),
        "exact_match": round(float(exact.sum()) / len(exact), 4),
    }


def eval_embeddings(X: torch.Tensor, Y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return _COS(X, Y).mean(), _MSE(X, Y)
