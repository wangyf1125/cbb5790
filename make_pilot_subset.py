from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ============================================================
# Defaults (can be overridden on the command line)
# ============================================================
DEFAULT_INPUT_DIR = Path(
    r"D:\yale_files\courses_cbb\02_Semester_Courses\CBB_5790\Mid&Final_Project\data_processed"
)
DEFAULT_OUTPUT_DIR = Path(
    r"D:\yale_files\courses_cbb\02_Semester_Courses\CBB_5790\Mid&Final_Project\data_processed_pilot"
)

CLIENTS = [
    "Client_0_Medicine",
    "Client_1_Surgery",
    "Client_2_Cardio",
    "Client_3_Others",
]

TARGET_SIZES = {
    "train": 1000,
    "val": 100,
    "test": 100,
}

BASE_SEED = 42


# ============================================================
# Utilities
# ============================================================
def reservoir_sample_jsonl(
    file_path: Path,
    k: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Reservoir-sample k records from a large JSONL file without loading
    the whole file into memory.

    Returns
    -------
    sampled_records : list[dict]
        Sampled JSON objects.
    total_count : int
        Total number of records seen in the file.
    """
    rng = random.Random(seed)
    reservoir: List[Dict[str, Any]] = []
    total_count = 0

    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            total_count += 1

            if total_count <= k:
                reservoir.append(record)
            else:
                j = rng.randint(1, total_count)
                if j <= k:
                    reservoir[j - 1] = record

    return reservoir, total_count


def write_jsonl(records: List[Dict[str, Any]], file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def shuffle_records(records: List[Dict[str, Any]], seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    copied = list(records)
    rng.shuffle(copied)
    return copied


# ============================================================
# Main
# ============================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reservoir-sample per-client pilot subsets from a federated JSONL dataset."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing <client>_<split>.jsonl files (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write pilot subsets and stats into (default: %(default)s)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    pilot_stats: Dict[str, Any] = {
        "base_seed": BASE_SEED,
        "target_sizes_per_client": TARGET_SIZES,
        "clients": {},
        "centralized": {},
    }

    centralized_buffers: Dict[str, List[Dict[str, Any]]] = {
        "train": [],
        "val": [],
        "test": [],
    }

    for client_idx, client_name in enumerate(CLIENTS):
        pilot_stats["clients"][client_name] = {}

        for split_idx, split_name in enumerate(["train", "val", "test"]):
            target_k = TARGET_SIZES[split_name]
            in_file = input_dir / f"{client_name}_{split_name}.jsonl"

            if not in_file.exists():
                raise FileNotFoundError(f"Missing input file: {in_file}")

            # Derive a deterministic but different seed per file
            seed = BASE_SEED + client_idx * 100 + split_idx

            sampled_records, total_count = reservoir_sample_jsonl(
                file_path=in_file,
                k=target_k,
                seed=seed,
            )

            actual_k = len(sampled_records)
            out_file = output_dir / f"{client_name}_{split_name}.jsonl"
            write_jsonl(sampled_records, out_file)

            centralized_buffers[split_name].extend(sampled_records)

            pilot_stats["clients"][client_name][split_name] = {
                "source_file": str(in_file),
                "total_available": total_count,
                "requested": target_k,
                "sampled": actual_k,
                "output_file": str(out_file),
            }

            print(
                f"[OK] {client_name} {split_name}: sampled {actual_k} / {total_count}"
            )

    # ------------------------------------------------------------
    # Build centralized pooled pilot splits
    # ------------------------------------------------------------
    for split_idx, split_name in enumerate(["train", "val", "test"]):
        pooled = shuffle_records(
            centralized_buffers[split_name],
            seed=BASE_SEED + 1000 + split_idx,
        )

        out_file = output_dir / f"Centralized_{split_name}.jsonl"
        write_jsonl(pooled, out_file)

        pilot_stats["centralized"][split_name] = {
            "num_samples": len(pooled),
            "output_file": str(out_file),
        }

        print(f"[OK] Centralized {split_name}: {len(pooled)} samples")

    # ------------------------------------------------------------
    # Save pilot stats
    # ------------------------------------------------------------
    stats_file = output_dir / "pilot_stats.json"
    with stats_file.open("w", encoding="utf-8") as f:
        json.dump(pilot_stats, f, indent=4, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("Pilot subset generation finished.")
    print(f"Output directory: {output_dir}")
    print(f"Stats file: {stats_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
