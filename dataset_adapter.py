from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from .types import TrainConfig


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------
def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    """
    Read a JSONL file into a list of Python dictionaries.

    Args:
        path: Path to a .jsonl file.

    Returns:
        List of parsed JSON objects.

    Raises:
        ValueError: If any line is invalid JSON.
    """
    path = Path(path)
    rows: List[Dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_num} of {path}") from exc

    return rows


def sample_rows(
    rows: Sequence[Dict[str, Any]],
    max_examples: Optional[int] = None,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Optionally subsample rows for fast debugging / pilot runs.

    Args:
        rows: Input examples.
        max_examples: If None, keep all. Otherwise sample up to this many.
        seed: Random seed for reproducibility.
    """
    rows = list(rows)
    if max_examples is None or max_examples >= len(rows):
        return rows

    rng = random.Random(seed)
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    keep = sorted(indices[:max_examples])
    return [rows[i] for i in keep]


def filter_summarization_rows(
    rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Keep only valid summarization rows.

    Expected raw schema
    -------------------
    Required keys:
        - instruction
        - input
        - output
        - task == "summarization"
    """
    kept: List[Dict[str, Any]] = []

    for row in rows:
        if row.get("task") != "summarization":
            continue

        instruction = str(row.get("instruction", "")).strip()
        input_text = str(row.get("input", "")).strip()
        output_text = str(row.get("output", "")).strip()

        if not instruction or not input_text or not output_text:
            continue

        kept.append(row)

    return kept


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------
def build_chat_messages(
    instruction: str,
    input_text: str,
) -> List[Dict[str, str]]:
    """
    Build chat messages for Meta-Llama-3.1-Instruct.
    """
    user_content = instruction.strip()
    if input_text.strip():
        user_content = f"{user_content}\n\n{input_text.strip()}"

    return [
        {
            "role": "system",
            "content": "You are a helpful clinical summarization assistant.",
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]


def render_prompt_text(
    tokenizer: Any,
    instruction: str,
    input_text: str,
) -> str:
    """
    Render a chat-style prompt using tokenizer.apply_chat_template when
    available; otherwise fall back to a plain text instruction format.
    """
    messages = build_chat_messages(instruction, input_text)

    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    # Fallback: simple instruction-style prompt
    return (
        "System: You are a helpful clinical summarization assistant.\n\n"
        f"User: {instruction.strip()}\n\n{input_text.strip()}\n\n"
        "Assistant:"
    )


# ---------------------------------------------------------------------------
# Tokenization helpers
# ---------------------------------------------------------------------------
def _tokenize_prompt(
    tokenizer: Any,
    prompt_text: str,
    train_config: TrainConfig,
) -> List[int]:
    """
    Tokenize prompt text only, truncated to max_source_length.
    """
    enc = tokenizer(
        prompt_text,
        add_special_tokens=False,
        truncation=train_config.truncation,
        max_length=train_config.max_source_length,
        padding=False,
        return_attention_mask=False,
    )
    return list(enc["input_ids"])


def _tokenize_target(
    tokenizer: Any,
    target_text: str,
    train_config: TrainConfig,
) -> List[int]:
    """
    Tokenize target text only, truncated to max_target_length.

    We append EOS if available so the model learns a stopping boundary.
    """
    target_text = target_text.strip()
    if getattr(tokenizer, "eos_token", None):
        target_text = target_text + tokenizer.eos_token

    enc = tokenizer(
        target_text,
        add_special_tokens=False,
        truncation=train_config.truncation,
        max_length=train_config.max_target_length,
        padding=False,
        return_attention_mask=False,
    )
    return list(enc["input_ids"])


# ---------------------------------------------------------------------------
# Dataset classes
# ---------------------------------------------------------------------------
class SummarizationTrainDataset(Dataset):
    """
    Tokenized training dataset for supervised LoRA fine-tuning.

    Each item returns unpadded 1D tensors:
        - input_ids
        - attention_mask
        - labels

    Padding is done later by the train collator.
    """

    def __init__(
        self,
        rows: Sequence[Dict[str, Any]],
        tokenizer: Any,
        train_config: TrainConfig,
    ) -> None:
        self.rows = list(rows)
        self.tokenizer = tokenizer
        self.train_config = train_config

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]

        instruction = str(row["instruction"])
        input_text = str(row["input"])
        output_text = str(row["output"])

        prompt_text = render_prompt_text(
            tokenizer=self.tokenizer,
            instruction=instruction,
            input_text=input_text,
        )

        prompt_ids = _tokenize_prompt(
            tokenizer=self.tokenizer,
            prompt_text=prompt_text,
            train_config=self.train_config,
        )
        target_ids = _tokenize_target(
            tokenizer=self.tokenizer,
            target_text=output_text,
            train_config=self.train_config,
        )

        full_input_ids = prompt_ids + target_ids
        full_attention_mask = [1] * len(full_input_ids)
        full_labels = ([-100] * len(prompt_ids)) + target_ids

        return {
            "input_ids": torch.tensor(full_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(full_attention_mask, dtype=torch.long),
            "labels": torch.tensor(full_labels, dtype=torch.long),
        }


class SummarizationEvalDataset(Dataset):
    """
    Generation-only evaluation dataset.

    Each item returns unpadded prompt tensors plus the gold reference string:
        - input_ids        (prompt tokens, to be left-padded by the collator)
        - attention_mask   (all ones, same length as input_ids)
        - reference

    No teacher-forced ``labels`` are produced: evaluation is ROUGE over
    generated summaries, so carrying a full prompt+target sequence would
    just waste memory.
    """

    def __init__(
        self,
        rows: Sequence[Dict[str, Any]],
        tokenizer: Any,
        train_config: TrainConfig,
    ) -> None:
        self.rows = list(rows)
        self.tokenizer = tokenizer
        self.train_config = train_config

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]

        instruction = str(row["instruction"])
        input_text = str(row["input"])
        output_text = str(row["output"])

        prompt_text = render_prompt_text(
            tokenizer=self.tokenizer,
            instruction=instruction,
            input_text=input_text,
        )

        prompt_ids = _tokenize_prompt(
            tokenizer=self.tokenizer,
            prompt_text=prompt_text,
            train_config=self.train_config,
        )
        prompt_attention_mask = [1] * len(prompt_ids)

        example: Dict[str, Any] = {
            "input_ids": torch.tensor(prompt_ids, dtype=torch.long),
            "attention_mask": torch.tensor(prompt_attention_mask, dtype=torch.long),
            "reference": output_text.strip(),
        }

        # Keep optional metadata for debugging / error analysis
        for key in ["subject_id", "hadm_id", "client_node", "curr_service", "split"]:
            if key in row:
                example[key] = row.get(key)

        return example


# ---------------------------------------------------------------------------
# Collators
# ---------------------------------------------------------------------------
def _left_pad_tensor_list(
    tensors: Sequence[torch.Tensor],
    pad_value: int,
) -> torch.Tensor:
    """
    Left-pad a list of 1D tensors into a [B, L] tensor.
    """
    max_len = max(t.size(0) for t in tensors)
    batch_size = len(tensors)

    out = torch.full(
        (batch_size, max_len),
        fill_value=pad_value,
        dtype=tensors[0].dtype,
    )

    for i, t in enumerate(tensors):
        out[i, max_len - t.size(0) :] = t

    return out


class TrainCollator:
    """
    Right-padding collator for supervised training batches.
    """

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def __call__(self, batch: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids = [x["input_ids"] for x in batch]
        attention_mask = [x["attention_mask"] for x in batch]
        labels = [x["labels"] for x in batch]

        return {
            "input_ids": pad_sequence(
                input_ids,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id,
            ),
            "attention_mask": pad_sequence(
                attention_mask,
                batch_first=True,
                padding_value=0,
            ),
            "labels": pad_sequence(
                labels,
                batch_first=True,
                padding_value=-100,
            ),
        }


class EvalCollator:
    """
    Left-padding collator for generation-only evaluation batches.
    """

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def __call__(self, batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        input_ids = [x["input_ids"] for x in batch]
        attention_mask = [x["attention_mask"] for x in batch]
        references = [x["reference"] for x in batch]

        collated: Dict[str, Any] = {
            "input_ids": _left_pad_tensor_list(
                input_ids,
                pad_value=self.tokenizer.pad_token_id,
            ),
            "attention_mask": _left_pad_tensor_list(
                attention_mask,
                pad_value=0,
            ),
            "reference": references,
        }

        # Preserve optional metadata if present
        for key in ["subject_id", "hadm_id", "client_node", "curr_service", "split"]:
            if key in batch[0]:
                collated[key] = [x.get(key) for x in batch]

        return collated


# ---------------------------------------------------------------------------
# Dataloader builders
# ---------------------------------------------------------------------------
def build_train_dataloader(
    jsonl_path: str | Path,
    tokenizer: Any,
    train_config: TrainConfig,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    max_examples: Optional[int] = None,
    seed: int = 42,
) -> DataLoader:
    """
    Build a training dataloader from a summarization JSONL file.
    """
    rows = read_jsonl(jsonl_path)
    rows = filter_summarization_rows(rows)
    rows = sample_rows(rows, max_examples=max_examples, seed=seed)

    dataset = SummarizationTrainDataset(
        rows=rows,
        tokenizer=tokenizer,
        train_config=train_config,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=TrainCollator(tokenizer),
    )


def build_eval_dataloader(
    jsonl_path: str | Path,
    tokenizer: Any,
    train_config: TrainConfig,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = True,
    max_examples: Optional[int] = None,
    seed: int = 42,
) -> DataLoader:
    """
    Build an evaluation/generation dataloader from a summarization JSONL file.
    """
    rows = read_jsonl(jsonl_path)
    rows = filter_summarization_rows(rows)
    rows = sample_rows(rows, max_examples=max_examples, seed=seed)

    dataset = SummarizationEvalDataset(
        rows=rows,
        tokenizer=tokenizer,
        train_config=train_config,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=EvalCollator(tokenizer),
    )


def build_client_dataloaders(
    data_dir: str | Path,
    client_name: str,
    tokenizer: Any,
    train_config: TrainConfig,
    train_batch_size: int = 1,
    eval_batch_size: int = 1,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> Dict[str, DataLoader]:
    """
    Build train / val / test dataloaders for one client.

    Expected files:
        <data_dir>/<client_name>_train.jsonl
        <data_dir>/<client_name>_val.jsonl
        <data_dir>/<client_name>_test.jsonl
    """
    data_dir = Path(data_dir)

    train_path = data_dir / f"{client_name}_train.jsonl"
    val_path = data_dir / f"{client_name}_val.jsonl"
    test_path = data_dir / f"{client_name}_test.jsonl"

    return {
        "train": build_train_dataloader(
            jsonl_path=train_path,
            tokenizer=tokenizer,
            train_config=train_config,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "val": build_eval_dataloader(
            jsonl_path=val_path,
            tokenizer=tokenizer,
            train_config=train_config,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "test": build_eval_dataloader(
            jsonl_path=test_path,
            tokenizer=tokenizer,
            train_config=train_config,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
    }


def build_centralized_dataloaders(
    data_dir: str | Path,
    tokenizer: Any,
    train_config: TrainConfig,
    train_batch_size: int = 1,
    eval_batch_size: int = 1,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> Dict[str, DataLoader]:
    """
    Build train / val / test dataloaders for centralized pilot files.

    Expected files:
        <data_dir>/Centralized_train.jsonl
        <data_dir>/Centralized_val.jsonl
        <data_dir>/Centralized_test.jsonl
    """
    data_dir = Path(data_dir)

    train_path = data_dir / "Centralized_train.jsonl"
    val_path = data_dir / "Centralized_val.jsonl"
    test_path = data_dir / "Centralized_test.jsonl"

    return {
        "train": build_train_dataloader(
            jsonl_path=train_path,
            tokenizer=tokenizer,
            train_config=train_config,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "val": build_eval_dataloader(
            jsonl_path=val_path,
            tokenizer=tokenizer,
            train_config=train_config,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "test": build_eval_dataloader(
            jsonl_path=test_path,
            tokenizer=tokenizer,
            train_config=train_config,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
    }
