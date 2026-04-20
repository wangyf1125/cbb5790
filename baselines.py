from __future__ import annotations

import time
from dataclasses import replace
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader

from .eval import compute_summarization_metrics
from .model_utils import load_base_model_and_tokenizer, setup_lora_model
from .train_utils import (
    evaluate_local_summarization,
    generate_summaries,
    run_local_training,
)
from .types import GenerationConfig, ModelConfig, TrainConfig


def _resolve_device(device: Optional[torch.device]) -> torch.device:
    if device is not None:
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _count_examples(dataloader: DataLoader) -> int:
    dataset = getattr(dataloader, "dataset", None)
    if dataset is not None:
        try:
            return int(len(dataset))
        except TypeError:
            pass
    try:
        return int(len(dataloader))
    except TypeError:
        return 0


def run_zero_shot_baseline(
    eval_dataloader: DataLoader,
    model_config: ModelConfig,
    generation_config: GenerationConfig,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Evaluate the base model with no fine-tuning.
    """
    device = _resolve_device(device)

    # Zero-shot is pure inference: we want KV cache on and no gradient
    # checkpointing, regardless of how the default ModelConfig is wired for
    # training.
    inference_config = replace(
        model_config,
        use_gradient_checkpointing=False,
        use_cache_during_training=True,
    )

    model, tokenizer = load_base_model_and_tokenizer(inference_config)
    model.to(device)

    start = time.time()
    predictions, references = generate_summaries(
        model=model,
        tokenizer=tokenizer,
        dataloader=eval_dataloader,
        generation_config=generation_config,
        device=device,
    )
    gen_seconds = time.time() - start

    metrics = compute_summarization_metrics(
        predictions=predictions,
        references=references,
    )

    return {
        "baseline": "zero_shot",
        "num_eval_examples": _count_examples(eval_dataloader),
        "generation_seconds": gen_seconds,
        "metrics": metrics,
    }


def run_centralized_lora_baseline(
    train_dataloader: DataLoader,
    eval_dataloader: DataLoader,
    model_config: ModelConfig,
    train_config: TrainConfig,
    generation_config: GenerationConfig,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Fine-tune LoRA on pooled training data, then evaluate.
    """
    device = _resolve_device(device)
    model, tokenizer = setup_lora_model(model_config)
    model.to(device)

    train_start = time.time()
    train_metrics = run_local_training(
        model=model,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        train_config=train_config,
        device=device,
    )
    train_seconds = time.time() - train_start

    eval_start = time.time()
    eval_metrics = evaluate_local_summarization(
        model=model,
        tokenizer=tokenizer,
        val_dataloader=eval_dataloader,
        generation_config=generation_config,
        device=device,
    )
    eval_seconds = time.time() - eval_start

    return {
        "baseline": "centralized_lora",
        "num_train_examples": _count_examples(train_dataloader),
        "num_eval_examples": _count_examples(eval_dataloader),
        "train_seconds": train_seconds,
        "eval_seconds": eval_seconds,
        "train_metrics": train_metrics,
        "metrics": eval_metrics,
    }
