from __future__ import annotations

import argparse
import csv
import gc
import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import flwr as fl
import torch
from flwr.common import Metrics, NDArrays, Scalar, ndarrays_to_parameters, parameters_to_ndarrays

from .client import MedicalFLClient
from .dataset_adapter import build_centralized_dataloaders, build_client_dataloaders
from .lora_state import adapter_size_mb_from_arrays, get_lora_parameters, set_lora_parameters
from .model_utils import setup_lora_model
from .baselines import run_zero_shot_baseline
from .train_utils import evaluate_local_summarization, run_local_training
from .types import GenerationConfig, ModelConfig, TrainConfig


CLIENTS = [
    "Client_0_Medicine",
    "Client_1_Surgery",
    "Client_2_Cardio",
    "Client_3_Others",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the B-person Flower federated LoRA pilot and log quality/system metrics."
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("federated_outputs"))
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--num-cpus-per-client", type=float, default=4.0)
    parser.add_argument("--num-gpus-per-client", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-final-test",
        action="store_true",
        help="Skip final global test-set evaluation. Useful for quick smoke tests.",
    )
    parser.add_argument(
        "--run-zero-shot",
        action="store_true",
        help="Evaluate the unfine-tuned base model on the centralized test split before FL.",
    )
    return parser.parse_args()


def _weighted_average(metrics: List[Tuple[int, Dict[str, Any]]], key: str) -> float:
    numer = 0.0
    denom = 0
    for num_examples, m in metrics:
        value = m.get(key)
        if isinstance(value, (int, float)):
            numer += float(value) * int(num_examples)
            denom += int(num_examples)
    return numer / denom if denom > 0 else 0.0


def _simple_mean(metrics: List[Tuple[int, Dict[str, Any]]], key: str) -> float:
    values = [float(m[key]) for _, m in metrics if isinstance(m.get(key), (int, float))]
    return sum(values) / len(values) if values else 0.0


def _simple_max(metrics: List[Tuple[int, Dict[str, Any]]], key: str) -> float:
    values = [float(m[key]) for _, m in metrics if isinstance(m.get(key), (int, float))]
    return max(values) if values else 0.0


def _simple_sum(metrics: List[Tuple[int, Dict[str, Any]]], key: str) -> float:
    values = [float(m[key]) for _, m in metrics if isinstance(m.get(key), (int, float))]
    return sum(values) if values else 0.0


class RoundLogger:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.records: List[Dict[str, Any]] = []
        self.round_wallclock_seconds: Dict[int, float] = {}
        self._round_start_times: Dict[int, float] = {}
        self._pending_fit: Dict[Tuple[int, str], Dict[str, Any]] = {}

    def begin_round(self, server_round: int) -> None:
        self._round_start_times[server_round] = time.time()

    def record_fit(self, server_round: int, metrics: List[Tuple[int, Metrics]]) -> Metrics:
        typed = [(int(n), dict(m)) for n, m in metrics]
        for num_examples, m in typed:
            client_id = str(m.get("client_id", "unknown"))
            self._pending_fit[(server_round, client_id)] = {
                "round": server_round,
                "client_id": client_id,
                "num_train_examples": int(num_examples),
                "train_loss": float(m.get("train_loss", 0.0)),
                "fit_seconds": float(m.get("fit_seconds", 0.0)),
                "adapter_size_mb": float(m.get("adapter_size_mb", 0.0)),
                "one_way_comm_mb": float(m.get("one_way_comm_mb", 0.0)),
                "round_trip_comm_mb": float(m.get("round_trip_comm_mb", 0.0)),
            }

        return {
            "train_loss": _weighted_average(typed, "train_loss"),
            "fit_seconds_mean": _simple_mean(typed, "fit_seconds"),
            "fit_seconds_max": _simple_max(typed, "fit_seconds"),
            "adapter_size_mb_mean": _simple_mean(typed, "adapter_size_mb"),
            "total_one_way_comm_mb": _simple_sum(typed, "one_way_comm_mb"),
            "total_round_trip_comm_mb": _simple_sum(typed, "round_trip_comm_mb"),
            "num_participating_clients": float(len(typed)),
            "num_fit_examples": float(sum(n for n, _ in typed)),
        }

    def record_evaluate(self, server_round: int, metrics: List[Tuple[int, Metrics]]) -> Metrics:
        typed = [(int(n), dict(m)) for n, m in metrics]
        start = self._round_start_times.get(server_round)
        if start is not None:
            self.round_wallclock_seconds[server_round] = time.time() - start

        for num_examples, m in typed:
            client_id = str(m.get("client_id", "unknown"))
            fit_part = self._pending_fit.get((server_round, client_id), {})
            self.records.append(
                {
                    **fit_part,
                    "round": server_round,
                    "client_id": client_id,
                    "num_eval_examples": int(num_examples),
                    "rouge1": float(m.get("rouge1", 0.0)),
                    "rouge2": float(m.get("rouge2", 0.0)),
                    "rougeL": float(m.get("rougeL", 0.0)),
                    "flower_surrogate_loss": float(m.get("flower_surrogate_loss", 0.0)),
                    "eval_seconds": float(m.get("eval_seconds", 0.0)),
                    "round_wallclock_seconds": float(
                        self.round_wallclock_seconds.get(server_round, 0.0)
                    ),
                }
            )

        return {
            "rouge1": _weighted_average(typed, "rouge1"),
            "rouge2": _weighted_average(typed, "rouge2"),
            "rougeL": _weighted_average(typed, "rougeL"),
            "eval_seconds_mean": _simple_mean(typed, "eval_seconds"),
            "eval_seconds_max": _simple_max(typed, "eval_seconds"),
            "flower_surrogate_loss": _weighted_average(typed, "flower_surrogate_loss"),
            "num_eval_examples": float(sum(n for n, _ in typed)),
        }

    def write_round_logs(self) -> Dict[str, Any]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "records": self.records,
            "round_wallclock_seconds": {
                str(k): float(v) for k, v in self.round_wallclock_seconds.items()
            },
            "total_one_way_comm_mb": float(
                sum(r.get("one_way_comm_mb", 0.0) for r in self.records)
            ),
            "total_round_trip_comm_mb": float(
                sum(r.get("round_trip_comm_mb", 0.0) for r in self.records)
            ),
        }

        json_path = self.output_dir / "federated_round_metrics.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        csv_path = self.output_dir / "federated_round_metrics.csv"
        if self.records:
            fieldnames = sorted({k for r in self.records for k in r.keys()})
            with csv_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.records)

        return summary


class TrackingFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, round_logger: RoundLogger, local_epochs: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.round_logger = round_logger
        self.local_epochs = local_epochs
        self.final_parameters = None

    def configure_fit(self, server_round, parameters, client_manager):
        self.round_logger.begin_round(server_round)
        return super().configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, _ = super().aggregate_fit(server_round, results, failures)
        if aggregated_parameters is not None:
            self.final_parameters = aggregated_parameters

        metrics = [(fit_res.num_examples, fit_res.metrics) for _, fit_res in results]
        aggregated_metrics = self.round_logger.record_fit(server_round, metrics)
        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(self, server_round, results, failures):
        aggregated_loss, _ = super().aggregate_evaluate(server_round, results, failures)
        metrics = [(eval_res.num_examples, eval_res.metrics) for _, eval_res in results]
        aggregated_metrics = self.round_logger.record_evaluate(server_round, metrics)
        return aggregated_loss, aggregated_metrics


def fit_config_fn(local_epochs: int) -> Callable[[int], Dict[str, Scalar]]:
    def inner(server_round: int) -> Dict[str, Scalar]:
        return {"local_epochs": int(local_epochs), "server_round": int(server_round)}

    return inner


def make_client_fn(
    data_dir: Path,
    model_config: ModelConfig,
    train_config: TrainConfig,
    generation_config: GenerationConfig,
    train_batch_size: int,
    eval_batch_size: int,
) -> Callable[[str], fl.client.Client]:
    def client_fn(cid: str) -> fl.client.Client:
        client_idx = int(cid)
        client_name = CLIENTS[client_idx]

        model, tokenizer = setup_lora_model(model_config)
        loaders = build_client_dataloaders(
            data_dir=data_dir,
            client_name=client_name,
            tokenizer=tokenizer,
            train_config=train_config,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
        )

        client = MedicalFLClient(
            model=model,
            tokenizer=tokenizer,
            train_dataloader=loaders["train"],
            val_dataloader=loaders["val"],
            train_fn=run_local_training,
            eval_fn=evaluate_local_summarization,
            client_id=client_name,
            train_config=train_config,
            generation_config=generation_config,
        )
        return client.to_client()

    return client_fn


def evaluate_final_global_model(
    data_dir: Path,
    final_parameters_ndarrays: NDArrays,
    model_config: ModelConfig,
    train_config: TrainConfig,
    generation_config: GenerationConfig,
    eval_batch_size: int,
) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = setup_lora_model(model_config)
    model.to(device)
    set_lora_parameters(model, final_parameters_ndarrays)

    per_client_test: Dict[str, Dict[str, float]] = {}
    total_examples = 0.0
    accum = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    for client_name in CLIENTS:
        loaders = build_client_dataloaders(
            data_dir=data_dir,
            client_name=client_name,
            tokenizer=tokenizer,
            train_config=train_config,
            train_batch_size=1,
            eval_batch_size=eval_batch_size,
        )
        metrics = evaluate_local_summarization(
            model=model,
            tokenizer=tokenizer,
            val_dataloader=loaders["test"],
            generation_config=generation_config,
            device=device,
        )
        per_client_test[client_name] = metrics

        n = float(metrics.get("num_eval_examples", 0.0))
        total_examples += n
        for key in accum:
            accum[key] += float(metrics.get(key, 0.0)) * n

    weighted_summary = {
        key: (value / total_examples if total_examples > 0 else 0.0)
        for key, value in accum.items()
    }
    weighted_summary["num_eval_examples"] = total_examples

    centralized_test: Optional[Dict[str, float]]
    try:
        centralized_loaders = build_centralized_dataloaders(
            data_dir=data_dir,
            tokenizer=tokenizer,
            train_config=train_config,
            train_batch_size=1,
            eval_batch_size=eval_batch_size,
        )
        centralized_test = evaluate_local_summarization(
            model=model,
            tokenizer=tokenizer,
            val_dataloader=centralized_loaders["test"],
            generation_config=generation_config,
            device=device,
        )
    except FileNotFoundError:
        centralized_test = None

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return {
        "per_client_test": per_client_test,
        "weighted_test_summary": weighted_summary,
        "centralized_test": centralized_test,
    }


def evaluate_zero_shot_baseline(
    data_dir: Path,
    model_config: ModelConfig,
    train_config: TrainConfig,
    generation_config: GenerationConfig,
    eval_batch_size: int,
) -> Dict[str, Any]:
    model, tokenizer = setup_lora_model(model_config)
    loaders = build_centralized_dataloaders(
        data_dir=data_dir,
        tokenizer=tokenizer,
        train_config=train_config,
        train_batch_size=1,
        eval_batch_size=eval_batch_size,
    )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return run_zero_shot_baseline(
        eval_dataloader=loaders["test"],
        model_config=model_config,
        generation_config=generation_config,
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model_config = ModelConfig()
    train_config = TrainConfig(num_epochs=args.local_epochs)
    generation_config = GenerationConfig()

    zero_shot_results = None
    if args.run_zero_shot:
        print("[runner] running zero-shot centralized test baseline")
        zero_shot_results = evaluate_zero_shot_baseline(
            data_dir=args.data_dir,
            model_config=model_config,
            train_config=train_config,
            generation_config=generation_config,
            eval_batch_size=args.eval_batch_size,
        )
        print("[runner] zero-shot centralized test summary:")
        print(json.dumps(zero_shot_results, indent=2, ensure_ascii=False))

    print("[runner] building initial LoRA adapter")
    init_model, _ = setup_lora_model(model_config)
    initial_ndarrays = get_lora_parameters(init_model)
    initial_parameters = ndarrays_to_parameters(initial_ndarrays)
    init_adapter_mb = adapter_size_mb_from_arrays(initial_ndarrays)
    print(f"[runner] adapter payload: {init_adapter_mb:.3f} MiB one-way")

    del init_model, initial_ndarrays
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    round_logger = RoundLogger(args.output_dir)
    strategy = TrackingFedAvg(
        round_logger=round_logger,
        local_epochs=args.local_epochs,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=len(CLIENTS),
        min_evaluate_clients=len(CLIENTS),
        min_available_clients=len(CLIENTS),
        on_fit_config_fn=fit_config_fn(args.local_epochs),
        initial_parameters=initial_parameters,
    )

    client_fn = make_client_fn(
        data_dir=args.data_dir,
        model_config=model_config,
        train_config=train_config,
        generation_config=generation_config,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
    )

    simulation_start = time.time()
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=len(CLIENTS),
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
        client_resources={
            "num_cpus": args.num_cpus_per_client,
            "num_gpus": args.num_gpus_per_client,
        },
        ray_init_args={
            "ignore_reinit_error": True,
            "include_dashboard": False,
        },
    )
    total_training_time_seconds = time.time() - simulation_start

    round_summary = round_logger.write_round_logs()

    final_test_results = None
    if not args.skip_final_test:
        if strategy.final_parameters is None:
            raise RuntimeError("No final aggregated parameters were captured.")
        final_test_results = evaluate_final_global_model(
            data_dir=args.data_dir,
            final_parameters_ndarrays=parameters_to_ndarrays(strategy.final_parameters),
            model_config=model_config,
            train_config=train_config,
            generation_config=generation_config,
            eval_batch_size=args.eval_batch_size,
        )

    result_payload = {
        "config": {
            "data_dir": str(args.data_dir),
            "output_dir": str(args.output_dir),
            "rounds": args.rounds,
            "local_epochs": args.local_epochs,
            "train_batch_size": args.train_batch_size,
            "eval_batch_size": args.eval_batch_size,
            "num_clients": len(CLIENTS),
            "num_cpus_per_client": args.num_cpus_per_client,
            "num_gpus_per_client": args.num_gpus_per_client,
            "seed": args.seed,
            "skip_final_test": args.skip_final_test,
            "run_zero_shot": args.run_zero_shot,
        },
        "zero_shot_baseline": zero_shot_results,
        "system_efficiency": {
            "total_training_time_seconds": float(total_training_time_seconds),
            **round_summary,
        },
        "history": {
            "losses_distributed": getattr(history, "losses_distributed", []),
            "losses_centralized": getattr(history, "losses_centralized", []),
            "metrics_distributed_fit": getattr(history, "metrics_distributed_fit", {}),
            "metrics_distributed": getattr(history, "metrics_distributed", {}),
            "metrics_centralized": getattr(history, "metrics_centralized", {}),
        },
        "final_test_results": final_test_results,
    }

    out_file = args.output_dir / "federated_pilot_results.json"
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(result_payload, f, indent=2, ensure_ascii=False)

    print("=" * 70)
    print("Federated pilot finished.")
    print(f"Saved results to: {out_file}")
    print(f"Saved per-round metrics to: {args.output_dir / 'federated_round_metrics.csv'}")
    print(f"Total training time (s): {total_training_time_seconds:.2f}")
    if final_test_results is not None:
        print("Final weighted test summary:")
        print(json.dumps(final_test_results["weighted_test_summary"], indent=2))
    print("=" * 70)


if __name__ == "__main__":
    main()
