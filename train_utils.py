from __future__ import annotations

import gc
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader

from .eval import compute_summarization_metrics
from .types import GenerationConfig, TrainConfig


def move_batch_to_device(
    batch: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    """
    Move all tensor values in a batch to the given device.
    """
    moved: Dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def train_one_client_epoch(
    model: torch.nn.Module,
    tokenizer: Any,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float = 1.0,
    logging_steps: int = 10,
) -> Dict[str, float]:
    """
    Train for one full epoch.
    """
    del tokenizer
    model.train()
    model.to(device)

    running_loss = 0.0
    num_batches = 0
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(dataloader):
        batch = move_batch_to_device(batch, device)

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            labels=batch["labels"],
        )
        loss = outputs.loss
        (loss / max(1, gradient_accumulation_steps)).backward()

        running_loss += float(loss.detach().item())
        num_batches += 1

        is_accum_step = ((step + 1) % gradient_accumulation_steps) == 0
        is_last = (step + 1) == len(dataloader)

        if is_accum_step or is_last:
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=max_grad_norm,
                )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if logging_steps > 0 and (step + 1) % logging_steps == 0:
            avg = running_loss / max(1, num_batches)
            print(f"[train] step={step + 1:>5d} avg_loss={avg:.4f}")

    avg_loss = running_loss / max(1, num_batches)
    return {"train_loss": float(avg_loss)}


def run_local_training(
    model: torch.nn.Module,
    tokenizer: Any,
    train_dataloader: DataLoader,
    train_config: TrainConfig,
    device: torch.device,
) -> Dict[str, float]:
    """
    Run local LoRA fine-tuning for train_config.num_epochs epochs.
    """
    model.to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )

    last_metrics: Dict[str, float] = {"train_loss": float("nan")}
    for epoch in range(train_config.num_epochs):
        print(f"[train] starting epoch {epoch + 1}/{train_config.num_epochs}")
        last_metrics = train_one_client_epoch(
            model=model,
            tokenizer=tokenizer,
            dataloader=train_dataloader,
            optimizer=optimizer,
            device=device,
            gradient_accumulation_steps=train_config.gradient_accumulation_steps,
            max_grad_norm=train_config.max_grad_norm,
            logging_steps=train_config.logging_steps,
        )
        print(
            f"[train] epoch {epoch + 1} done | "
            f"avg_loss={last_metrics['train_loss']:.4f}"
        )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    last_metrics["num_epochs"] = float(train_config.num_epochs)
    return last_metrics


@torch.no_grad()
def generate_summaries(
    model: torch.nn.Module,
    tokenizer: Any,
    dataloader: DataLoader,
    generation_config: GenerationConfig,
    device: torch.device,
) -> Tuple[List[str], List[str]]:
    """
    Generate summaries for all examples in dataloader.

    Expected eval batch keys:
    - input_ids
    - attention_mask
    - reference
    """
    model.eval()
    model.to(device)

    original_padding_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"

    original_use_cache = getattr(model.config, "use_cache", False)
    model.config.use_cache = True

    predictions: List[str] = []
    references: List[str] = []

    try:
        for batch in dataloader:
            raw_refs = batch.get("reference", None)
            if raw_refs is None:
                raise KeyError(
                    "generate_summaries expects a 'reference' field in each eval batch."
                )

            batch = move_batch_to_device(batch, device)

            input_ids = batch["input_ids"]
            attention_mask = batch.get("attention_mask")

            gen_out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=generation_config.max_new_tokens,
                num_beams=generation_config.num_beams,
                do_sample=generation_config.do_sample,
                temperature=generation_config.temperature,
                top_p=generation_config.top_p,
                repetition_penalty=generation_config.repetition_penalty,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            prompt_len = input_ids.shape[1]
            new_tokens = gen_out[:, prompt_len:]
            decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

            predictions.extend([t.strip() for t in decoded])
            references.extend([str(r).strip() for r in raw_refs])

    finally:
        tokenizer.padding_side = original_padding_side
        model.config.use_cache = original_use_cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return predictions, references


def evaluate_local_summarization(
    model: torch.nn.Module,
    tokenizer: Any,
    val_dataloader: DataLoader,
    generation_config: GenerationConfig,
    device: torch.device,
) -> Dict[str, float]:
    """
    Standard local summarization evaluation used by both baselines and FL client.

    Generation + ROUGE only.
    """
    predictions, references = generate_summaries(
        model=model,
        tokenizer=tokenizer,
        dataloader=val_dataloader,
        generation_config=generation_config,
        device=device,
    )

    metrics = compute_summarization_metrics(
        predictions=predictions,
        references=references,
    )
    metrics["num_eval_examples"] = float(len(references))
    return metrics
