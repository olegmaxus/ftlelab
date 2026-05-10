import torch
import torch.nn as nn
from dataclasses import dataclass

__all__ = [
    "LOSS_MAP",
    "OPTIMIZER_MAP",
    "PARAM_MODULES",
    "TrainConfig"
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
class TrainConfig:
    lr: float = 1e-3
    epochs: int = 200
    batch_size: int = 256
    weight_decay: float = 0.0
    optimizer: str = "adam"
    momentum: float = 0.0

    # Task setup
    task: str = "binary"   # "binary", "multiclass", "autoencoder", "vae"
    loss: str = "mse"      # binary: mse / bce / bce_logits ; multiclass: ce ; ae/vae: mse / bce / bce_logits
    beta: float = 1.0      # KL weight for VAE
    target_from_input: bool = False   # if True, use x as target (autoencoders)

    save_dir: str = "checkpoints"
    model_name: str = "0"
    print_every: int = 10

    # Freezing
    train_only_output: bool = False
    train_last_n_param_modules: int = 0
    train_param_names: tuple = ()
    freeze_param_names: tuple = ()
    freeze_regex: str = ""
    train_module_names: tuple = ()    # e.g. ("decoder",) or ("head",)
    freeze_module_names: tuple = ()   # e.g. ("encoder",)