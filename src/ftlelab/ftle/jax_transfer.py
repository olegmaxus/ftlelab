from typing import Any, Dict

import jax.numpy as jnp
import torch
from torch import nn

from ..models.autoencoder import AutoEncoder


def _linear_layers_from_sequential(module: nn.Sequential) -> list[Dict[str, Any]]:
    layers = []
    for m in module:
        if isinstance(m, nn.Linear):
            W = m.weight.detach().cpu().numpy()
            b = m.bias.detach().cpu().numpy()
            layers.append(
                {
                    "W": jnp.asarray(W),
                    "b": jnp.asarray(b),
                }
            )
    return layers


def pytorch_dense_to_jax_params(model: nn.Module) -> Dict[str, Any]:
    """
    Convert a ftlelab.models.dense.DenseNet instance to a JAX params dict
    for use with dense_forward / dense_hidden_k.

    Assumes model.net is a Sequential of:
      Linear, activation, (Dropout), ..., Linear, activation
    and all Linear layers are the ones you want to mirror in order.
    """
    return {"layers": _linear_layers_from_sequential(model.net)}


def pytorch_autoencoder_to_jax_params(model: AutoEncoder) -> Dict[str, Any]:
    """
    Convert a ftlelab.models.autoencoder.AutoEncoder to a JAX params dict.
    """
    if not isinstance(model, AutoEncoder):
        raise TypeError("Expected an AutoEncoder instance.")

    encoder_layers = _linear_layers_from_sequential(model.encoder)
    decoder_layers = _linear_layers_from_sequential(model.decoder)

    return {
        "encoder_layers": encoder_layers,
        "decoder_layers": decoder_layers,
        "encoder_hidden_depth": len(model.encoder_activation_modules),
        "decoder_hidden_depth": len(model.decoder_dims),
    }
