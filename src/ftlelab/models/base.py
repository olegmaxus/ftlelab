import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Iterable, Sequence


# ============================================================
# Helpers
# ============================================================


ACTS = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "leaky_relu": nn.LeakyReLU,
    "sigmoid": nn.Sigmoid,
    "softplus": nn.Softplus,
    "identity": nn.Identity
}

def make_activation(name: str, negative_slope: float = 0.01) -> nn.Module:
    name = name.lower()
    if name not in ACTS:
        raise ValueError(f"Unknown activation: {name}. Available: {list(ACTS.keys())}")
    if name == "leaky_relu":
        return nn.LeakyReLU(negative_slope=negative_slope)
    return ACTS[name]()


def _fan_in_from_weight(weight: torch.Tensor) -> int:
    """
    Generic fan_in for Linear / Conv / ConvTranspose.
    Uses PyTorch's private helper for correctness across layer types.
    """
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
    return fan_in


# ============================================================
# Base class
# ============================================================

class BaseNet(nn.Module):
    """
    Shared utilities for:
      - initialization
      - prediction
      - feature extraction hooks for FTLE
    negative_slope 
    """

    def __init__(self,
                 init_method: str | None = "paper",
                 activation_name: str = "tanh",
                 output_activation_name: str = "identity",
                 negative_slope: float = 0.01):

        super().__init__()

        self.init_method = init_method.lower() if init_method else None

        self.activation_name = activation_name.lower()
        self.output_activation_name = output_activation_name.lower()
        self.negative_slope = negative_slope

    def apply_initialization(self):
        if self.init_method:
            self.apply(self._initialize_weights)

    def _initialize_weights(self, module: nn.Module):
        if not isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            return

        if self.init_method == "paper":
            fan_in = _fan_in_from_weight(module.weight)
            std = (1.0 / fan_in) ** 0.5
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

        elif self.init_method in {"glorot", "xavier"}:
            if self.activation_name == "leaky_relu":
                gain = nn.init.calculate_gain("leaky_relu", self.negative_slope)
            else:
                # relu / tanh / sigmoid / linear / etc.
                gain_name = self.activation_name if self.activation_name != "identity" else "linear"
                try:
                    gain = nn.init.calculate_gain(gain_name)
                except ValueError:
                    gain = 1.0
            nn.init.xavier_uniform_(module.weight, gain=gain)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

        elif self.init_method in {"he", "kaiming"}:
            if self.activation_name == "leaky_relu":
                nn.init.kaiming_uniform_(module.weight,
                                         a=self.negative_slope,
                                         nonlinearity="leaky_relu")
            else:
                # for GELU / Softplus, using ReLU-style He is a common approximation
                nonlinearity = "relu" if self.activation_name in {"relu", "gelu", "softplus"} else "linear"
                nn.init.kaiming_uniform_(module.weight, nonlinearity=nonlinearity)

            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

        else:
            raise ValueError(f"Unknown init_method: {self.init_method}")

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        y = self.forward(x)
        if isinstance(y, tuple):
            # For VAEs etc. we usually want the reconstruction
            y = y[0]

        if y.ndim >= 2 and y.shape[-1] == 1:
            return torch.where(y >= 0, 1.0, -1.0).squeeze(-1)
        return torch.argmax(y, dim=-1)

    def feature_from_module(self, x: torch.Tensor, target_module: nn.Module) -> torch.Tensor:
        """
        Generic feature extractor for FTLE experiments.
        Useful for CNNs / autoencoders / arbitrary architectures.
        """
        bag = {}

        def hook(_m, _inp, out):
            bag["feat"] = out

        handle = target_module.register_forward_hook(hook)
        try:
            _ = self.forward(x)
        finally:
            handle.remove()

        if "feat" not in bag:
            raise RuntimeError("Target module did not produce an output during forward().")
        return bag["feat"]


