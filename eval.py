from __future__ import annotations

from typing import Dict, List

import evaluate


_EMPTY_PLACEHOLDER = " "


def _clean_text(text: str) -> str:
    """Normalize whitespace and guard against empty/None outputs."""
    if text is None:
        return _EMPTY_PLACEHOLDER
    cleaned = " ".join(str(text).split()).strip()
    return cleaned if cleaned else _EMPTY_PLACEHOLDER


def compute_summarization_metrics(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    """Compute ROUGE-1, ROUGE-2, and ROUGE-L for summarization."""
    if len(predictions) != len(references):
        raise ValueError(
            f"predictions and references must have equal length, "
            f"got {len(predictions)} vs {len(references)}"
        )

    if len(predictions) == 0:
        return {
            "rouge1": 0.0,
            "rouge2": 0.0,
            "rougeL": 0.0,
        }

    clean_preds = [_clean_text(p) for p in predictions]
    clean_refs = [_clean_text(r) for r in references]

    rouge = evaluate.load("rouge")
    rouge_out = rouge.compute(
        predictions=clean_preds,
        references=clean_refs,
        use_stemmer=True,
    )

    return {
        "rouge1": float(rouge_out.get("rouge1", 0.0)),
        "rouge2": float(rouge_out.get("rouge2", 0.0)),
        "rougeL": float(rouge_out.get("rougeL", 0.0)),
    }
