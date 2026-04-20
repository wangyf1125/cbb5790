from __future__ import annotations

import gc
import time
from dataclasses import replace
from typing import Any, Callable, Dict, List, Tuple

import flwr as fl
import numpy as np
import torch
from torch.utils.data import DataLoader

from .lora_state import (
    adapter_size_mb_from_arrays,
    get_lora_parameters,
    set_lora_parameters,
)
from .types import GenerationConfig, TrainConfig


TrainFn = Callable[..., Dict[str, float]]
EvalFn = Callable[..., Dict[str, float]]


class MedicalFLClient(fl.client.NumPyClient):
    """
    Flower client that exchanges only LoRA adapter parameters.

    This client also logs lightweight system-efficiency metrics:
    - adapter_size_mb
    - one_way_comm_mb
    - round_trip_comm_mb
    - fit_seconds
    - eval_seconds
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        train_fn: TrainFn,
        eval_fn: EvalFn,
        client_id: str,
        train_config: TrainConfig,
        generation_config: GenerationConfig,
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.train_fn = train_fn
        self.eval_fn = eval_fn
        self.client_id = client_id
        self.train_config = train_config
        self.generation_config = generation_config
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model.to(self.device)

    # ------------------------------------------------------------------
    # Parameter exchange
    # ------------------------------------------------------------------
    def get_parameters(self, config: Dict[str, Any]) -> List[np.ndarray]:
        del config
        return get_lora_parameters(self.model)

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        set_lora_parameters(self.model, parameters)

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------
    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Any],
    ) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        self.set_parameters(parameters)

        local_epochs = int(config.get("local_epochs", self.train_config.num_epochs))
        effective_train_config = replace(self.train_config, num_epochs=local_epochs)

        start_time = time.time()
        try:
            train_metrics = self.train_fn(
                model=self.model,
                tokenizer=self.tokenizer,
                train_dataloader=self.train_dataloader,
                train_config=effective_train_config,
                device=self.device,
            )
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        fit_seconds = time.time() - start_time

        updated = self.get_parameters(config={})
        num_examples = _num_examples(self.train_dataloader)

        adapter_size_mb = adapter_size_mb_from_arrays(updated)
        metrics: Dict[str, Any] = {
            "client_id": self.client_id,
            "adapter_size_mb": adapter_size_mb,
            "one_way_comm_mb": adapter_size_mb,
            "round_trip_comm_mb": 2.0 * adapter_size_mb,
            "fit_seconds": float(fit_seconds),
            **{
                k: float(v)
                for k, v in train_metrics.items()
                if isinstance(v, (int, float))
            },
        }
        return updated, num_examples, metrics

    # ------------------------------------------------------------------
    # evaluate
    # ------------------------------------------------------------------
    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Any],
    ) -> Tuple[float, int, Dict[str, Any]]:
        del config
        self.set_parameters(parameters)

        start_time = time.time()
        try:
            eval_metrics = self.eval_fn(
                model=self.model,
                tokenizer=self.tokenizer,
                val_dataloader=self.val_dataloader,
                generation_config=self.generation_config,
                device=self.device,
            )
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        eval_seconds = time.time() - start_time

        rouge_l = float(eval_metrics.get("rougeL", 0.0))
        flower_loss = max(0.0, 1.0 - rouge_l)
        num_examples = _num_examples(self.val_dataloader)

        adapter_arrays = get_lora_parameters(self.model)
        adapter_size_mb = adapter_size_mb_from_arrays(adapter_arrays)
        metrics: Dict[str, Any] = {
            "client_id": self.client_id,
            "adapter_size_mb": adapter_size_mb,
            "one_way_comm_mb": adapter_size_mb,
            "round_trip_comm_mb": 2.0 * adapter_size_mb,
            "eval_seconds": float(eval_seconds),
            "flower_surrogate_loss": float(flower_loss),
            **{
                k: float(v)
                for k, v in eval_metrics.items()
                if isinstance(v, (int, float))
            },
        }
        return flower_loss, num_examples, metrics


def _num_examples(dataloader: DataLoader) -> int:
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
