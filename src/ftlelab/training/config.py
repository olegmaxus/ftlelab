import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Any, Optional

__all__ = [
    "LOSS_MAP",
    "OPTIMIZER_MAP",
    "PARAM_MODULES",
    "FTLEConfig",
    "TrainConfig",
]

LOSS_MAP = {
    "mse": nn.MSELoss,
    "bce": nn.BCELoss,                  # expects probabilities
    "bce_logits": nn.BCEWithLogitsLoss, # expects logits
    "ce": nn.CrossEntropyLoss,
}

OPTIMIZER_MAP = {
    "adam": torch.optim.Adam,
    "sgd": torch.optim.SGD,
    "adamw": torch.optim.AdamW,
}

PARAM_MODULES = (
    nn.Linear,
    nn.Conv1d, nn.Conv2d, nn.Conv3d,
    nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d
)

@dataclass
class FTLEConfig:
    model_type: str = "dense"
    activation: str = "tanh"
    output_activation: str = "tanh"

    # Grid of INPUT points: shape (N, d), numpy array
    grid_X: Any = None
    
    # What to compute:
    # 1) layers="all"     -> input->hidden_1..hidden_L and input->output
    # 2) layers=(1,3,"output") -> input->hidden_1, input->hidden_3, input->output
    layers: Any = "all"

    # Optional explicit transitions, e.g.
    # (("hidden_k", 2), ("hidden_k", 5)), (("hidden_k", 3), "output")
    layer_pairs: tuple = field(default_factory=tuple)

    batch_size: int = 1024
    dtype: str = "float32"
    enable_x64: bool = False
    exact_if_dim_le: int = 4
    max_steps: int = 50
    tol: float = 1e-6

    save_subdir: str = "ftle"
    save_format: str = "npy"   # jax -> npy for now


@dataclass
class TrainConfig:
    lr: float = 1e-3
    max_epochs: int = 200
    batch_size: int = 256
    weight_decay: float = 0.0
    optimizer: str = "adam"
    momentum: float = 0.0

    # Task setup
    task: str = "binary"   # "binary", "multiclass", "autoencoder", "vae"
    loss: str = "mse"      # binary: mse / bce / bce_logits ; multiclass: ce ; ae/vae: mse / bce / bce_logits
    beta: float = 1.0      # KL weight for VAE
    target_from_input: bool = False   # if True, use x as target (autoencoders)
    target_val_acc: float = 0.95

    save_dir: str = "checkpoints"
    model_name: str = "0"
    print_every: int = 10

    # Dynamic LEs
    compute_ftle: bool = False
    ftle_start: int = 1
    ftle_every: int = 5
    ftle_backend: str = "jax"   # jax / torch
    ftle_config: Optional[FTLEConfig] = None

    # Freezing
    train_only_output: bool = False
    train_last_n_param_modules: int = 0
    train_param_names: tuple = ()
    freeze_param_names: tuple = ()
    freeze_regex: str = ""
    train_module_names: tuple = ()    # e.g. ("decoder",) or ("head",)
    freeze_module_names: tuple = ()   # e.g. ("encoder",)