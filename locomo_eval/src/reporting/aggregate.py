"""Aggregate results across runs and compute per-category stats."""
import json
from pathlib import Path

import numpy as np

from ..config import CATEGORY_NAMES
from ..pipeline.evaluate import EvaluatedPrediction


def _safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _safe_std(values: list[float]) -> float:
    return float(np.std(values)) if len(values) > 1 else 0.0


def aggregate_runs(
    all_runs: list[list[EvaluatedPrediction]],
    *,
    adapter_name: str,
    answer_model: str,
    judge_model: str,
) -> dict:
    """Aggregate N runs into a single results dict."""
    # Flatten all predictions across runs; group by qa_id to average per question
    per_qa: dict[str, list[EvaluatedPrediction]] = {}
    for run in all_runs:
        for pred in run:
            per_qa.setdefault(pred.qa_id, []).append(pred)

    # Per-category aggregation
    cat_data: dict[int, dict] = {}
    for preds in per_qa.values():
        cat = preds[0].category
        if cat not in cat_data:
            cat_data[cat] = {"j": [], "f1": [], "bleu1": [], "n": 0}
        avg_j = _safe_mean([1.0 if p.judge_label == "CORRECT" else 0.0 for p in preds])
        avg_f1 = _safe_mean([p.f1 for p in preds])
        avg_b1 = _safe_mean([p.bleu1_score for p in preds])
        cat_data[cat]["j"].append(avg_j)
        cat_data[cat]["f1"].append(avg_f1)
        cat_data[cat]["bleu1"].append(avg_b1)
        cat_data[cat]["n"] += 1

    per_category = {}
    for cat, data in sorted(cat_data.items()):
        name = f"{cat}_{CATEGORY_NAMES.get(cat, 'unknown')}"
        per_category[name] = {
            "j_mean": _safe_mean(data["j"]),
            "j_std": _safe_std(data["j"]),
            "f1_mean": _safe_mean(data["f1"]),
            "bleu1_mean": _safe_mean(data["bleu1"]),
            "n": data["n"],
        }

    all_preds_flat = [p for run in all_runs for p in run]
    overall_j = _safe_mean([1.0 if p.judge_label == "CORRECT" else 0.0 for p in all_preds_flat])
    overall_f1 = _safe_mean([p.f1 for p in all_preds_flat])
    overall_b1 = _safe_mean([p.bleu1_score for p in all_preds_flat])

    tokens = {
        "avg_retrieval": _safe_mean([p.retrieval_tokens for p in all_preds_flat]),
        "avg_prompt": _safe_mean([p.prompt_tokens for p in all_preds_flat]),
        "avg_completion": _safe_mean([p.completion_tokens for p in all_preds_flat]),
    }
    latency = {
        "avg_search_ms": _safe_mean([p.search_latency_ms for p in all_preds_flat]),
        "avg_generation_ms": _safe_mean([p.generation_latency_ms for p in all_preds_flat]),
    }

    return {
        "adapter": adapter_name,
        "runs": len(all_runs),
        "answer_model": answer_model,
        "judge_model": judge_model,
        "filter": "categories_1_to_4",
        "n_questions": len(per_qa),
        "per_category": per_category,
        "overall": {"j_mean": overall_j, "f1_mean": overall_f1, "bleu1_mean": overall_b1},
        "tokens": tokens,
        "latency": latency,
    }
