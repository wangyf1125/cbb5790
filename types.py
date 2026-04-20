from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    """
    Configuration for the base causal LM and its LoRA adapter.
    """

    model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])

    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_bias: str = "none"

    preferred_dtype: str = "bfloat16"  # {"bfloat16", "float16", "float32"}
    use_gradient_checkpointing: bool = True
    trust_remote_code: bool = False
    low_cpu_mem_usage: bool = True
    use_cache_during_training: bool = False

    def __post_init__(self) -> None:
        if self.preferred_dtype not in {"bfloat16", "float16", "float32"}:
            raise ValueError(
                "preferred_dtype must be one of {'bfloat16', 'float16', 'float32'}"
            )
        if self.lora_r <= 0:
            raise ValueError("lora_r must be positive")
        if self.lora_alpha <= 0:
            raise ValueError("lora_alpha must be positive")
        if not (0.0 <= self.lora_dropout < 1.0):
            raise ValueError("lora_dropout must be in [0, 1)")
        if not self.target_modules:
            raise ValueError("target_modules must be a non-empty list")


@dataclass
class TrainConfig:
    """
    Configuration for local training and tokenization lengths.
    """

    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    num_epochs: int = 1
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1

    max_source_length: int = 2048
    max_target_length: int = 256
    truncation: bool = True

    logging_steps: int = 10

    def __post_init__(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be >= 0")
        if self.num_epochs <= 0:
            raise ValueError("num_epochs must be positive")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.max_source_length <= 0 or self.max_target_length <= 0:
            raise ValueError("max_source_length and max_target_length must be positive")


@dataclass
class GenerationConfig:
    """
    Configuration for summarization generation.
    """

    max_new_tokens: int = 256
    num_beams: int = 1
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    repetition_penalty: float = 1.0

    def __post_init__(self) -> None:
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self.num_beams <= 0:
            raise ValueError("num_beams must be positive")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if not (0.0 < self.top_p <= 1.0):
            raise ValueError("top_p must be in (0, 1]")
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be positive")


@dataclass
class FederatedConfig:
    """
    Optional configuration describing a federated run.
    """

    num_rounds: int = 3
    num_clients: int = 4
    client_id: Optional[str] = None
    server_address: str = "0.0.0.0:8080"
    local_epochs_per_round: int = 1

    def __post_init__(self) -> None:
        if self.num_rounds <= 0:
            raise ValueError("num_rounds must be positive")
        if self.num_clients <= 0:
            raise ValueError("num_clients must be positive")
        if self.local_epochs_per_round <= 0:
            raise ValueError("local_epochs_per_round must be positive")
