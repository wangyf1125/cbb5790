from __future__ import annotations

from typing import List

import numpy as np
import torch
from peft import get_peft_model_state_dict, set_peft_model_state_dict


def get_lora_parameter_names(model: torch.nn.Module) -> List[str]:
    """
    Return adapter state-dict keys in deterministic order.
    """
    adapter_state = get_peft_model_state_dict(model)
    return sorted(adapter_state.keys())


def get_lora_parameters(model: torch.nn.Module) -> List[np.ndarray]:
    """
    Extract adapter-only tensors as float32 NumPy arrays for Flower transport.

    Notes
    -----
    We explicitly cast to float32 before .numpy() because NumPy / Flower
    transport is more robust with float32 than with bf16 tensors.
    """
    adapter_state = get_peft_model_state_dict(model)
    names = sorted(adapter_state.keys())

    arrays: List[np.ndarray] = []
    for name in names:
        tensor = adapter_state[name]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"Expected torch.Tensor for adapter key {name!r}, "
                f"got {type(tensor).__name__}"
            )
        arrays.append(tensor.detach().to(torch.float32).cpu().numpy())

    return arrays


def set_lora_parameters(
    model: torch.nn.Module,
    parameters: List[np.ndarray],
) -> None:
    """
    Load incoming adapter arrays back into the current model.
    """
    current_state = get_peft_model_state_dict(model)
    names = sorted(current_state.keys())

    if len(parameters) != len(names):
        raise ValueError(
            f"Adapter length mismatch: received {len(parameters)} arrays "
            f"but model expects {len(names)}"
        )

    new_state = {}
    for name, array in zip(names, parameters):
        target = current_state[name]

        if tuple(array.shape) != tuple(target.shape):
            raise ValueError(
                f"Shape mismatch for {name!r}: "
                f"received {tuple(array.shape)} vs expected {tuple(target.shape)}"
            )

        tensor = torch.from_numpy(np.asarray(array))
        tensor = tensor.to(dtype=target.dtype, device=target.device)
        new_state[name] = tensor

    set_peft_model_state_dict(model, new_state)


def adapter_size_mb_from_arrays(arrays: List[np.ndarray]) -> float:
    """
    Compute adapter payload size in MiB from already-extracted NumPy arrays.

    Prefer this over ``get_adapter_size_mb`` when the caller has just pulled
    the adapter with ``get_lora_parameters`` — it avoids re-traversing the
    PEFT state dict and re-casting every tensor.
    """
    total_bytes = int(sum(arr.nbytes for arr in arrays))
    return float(total_bytes / (1024**2))


def get_adapter_size_bytes(model: torch.nn.Module) -> int:
    """
    Return the adapter-only payload size in bytes, based on the exact NumPy
    arrays that Flower would transmit.
    """
    arrays = get_lora_parameters(model)
    return int(sum(arr.nbytes for arr in arrays))


def get_adapter_size_mb(model: torch.nn.Module) -> float:
    """
    Return the adapter-only payload size in megabytes (MiB).
    """
    total_bytes = get_adapter_size_bytes(model)
    return float(total_bytes / (1024**2))


def get_round_trip_comm_mb(model: torch.nn.Module) -> float:
    """
    Approximate per-client round-trip communication in MiB.

    One FL round for a client usually includes:
    - one downlink of global adapter parameters from server to client
    - one uplink of updated adapter parameters from client to server

    So round-trip communication is approximately 2 x adapter_size_mb.
    """
    adapter_mb = get_adapter_size_mb(model)
    return float(2.0 * adapter_mb)
