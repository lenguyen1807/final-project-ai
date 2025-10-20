from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingConfig:
    """
    Configuration class for training parameters.
    """

    model_type: str
    run_name: str
    encoder_model_name: Optional[str] = None
    decoder_model_name: Optional[str] = None
    use_lora: bool = False
    compute_clinical: bool = False
    seed: int = 42
    gpu_id: str = "0"
    num_epochs: int = 10
    batch_size: int = 8
    eval_batch_size: int = 16
    lr: float = 2e-5
    warmup: float = 0.1
    weight_decay: float = 1e-4
    accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    use_amp: bool = True
