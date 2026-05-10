
import torch
import torch.nn as nn
from .dense import DenseNet
from .conv import ConvNet
from .autoencoder import AutoEncoder, VAE

def make_model(kind: str, **kwargs) -> nn.Module:
    kind = kind.lower()
    if kind == "dense":
        return DenseNet(**kwargs)
    if kind == "conv":
        return ConvNet(**kwargs)
    if kind == "autoencoder":
        return AutoEncoder(**kwargs)
    if kind == "vae":
        return VAE(**kwargs)
    raise ValueError(f"Unknown model kind: {kind}")
