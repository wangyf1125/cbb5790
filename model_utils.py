from __future__ import annotations

from typing import Tuple

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from .types import ModelConfig


def resolve_torch_dtype(preferred: str = "bfloat16") -> torch.dtype:
    """
    Resolve dtype with safe bf16 -> fp16 fallback on CUDA.
    On CPU, prefer float32 for stability.
    """
    if preferred not in {"bfloat16", "float16", "float32"}:
        raise ValueError("preferred must be one of {'bfloat16', 'float16', 'float32'}")

    if not torch.cuda.is_available():
        return torch.float32

    if preferred == "bfloat16":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    if preferred == "float16":
        return torch.float16
    return torch.float32


def print_trainable_parameters(model: torch.nn.Module) -> None:
    """
    Print trainable vs total parameters.
    """
    trainable = 0
    total = 0
    for param in model.parameters():
        n = param.numel()
        total += n
        if param.requires_grad:
            trainable += n

    pct = 100.0 * trainable / total if total > 0 else 0.0
    print(
        f"[model_utils] trainable params: {trainable:,} || "
        f"all params: {total:,} || trainable%: {pct:.4f}"
    )


def _enable_gradient_checkpointing(model: torch.nn.Module) -> None:
    """
    Enable gradient checkpointing robustly across HF versions.
    """
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            model.gradient_checkpointing_enable()


def load_base_model_and_tokenizer(
    model_config: ModelConfig,
) -> Tuple[torch.nn.Module, PreTrainedTokenizerBase]:
    """
    Load the base model and tokenizer without attaching LoRA.
    """
    dtype = resolve_torch_dtype(model_config.preferred_dtype)

    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name,
        trust_remote_code=model_config.trust_remote_code,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name,
        torch_dtype=dtype,
        trust_remote_code=model_config.trust_remote_code,
        low_cpu_mem_usage=model_config.low_cpu_mem_usage,
    )

    model.config.use_cache = model_config.use_cache_during_training

    if (
        getattr(model.config, "pad_token_id", None) is None
        and tokenizer.pad_token_id is not None
    ):
        model.config.pad_token_id = tokenizer.pad_token_id

    if model_config.use_gradient_checkpointing:
        _enable_gradient_checkpointing(model)

    return model, tokenizer


def setup_lora_model(
    model_config: ModelConfig,
) -> Tuple[PeftModel, PreTrainedTokenizerBase]:
    """
    Load the base model and attach LoRA adapters.
    """
    base_model, tokenizer = load_base_model_and_tokenizer(model_config)

    lora_config = LoraConfig(
        r=model_config.lora_r,
        lora_alpha=model_config.lora_alpha,
        lora_dropout=model_config.lora_dropout,
        target_modules=list(model_config.target_modules),
        bias=model_config.lora_bias,
        task_type=TaskType.CAUSAL_LM,
    )

    peft_model = get_peft_model(base_model, lora_config)
    print_trainable_parameters(peft_model)
    return peft_model, tokenizer
