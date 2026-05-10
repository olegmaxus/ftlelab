from typing import Any, Dict

import jax.numpy as jnp
import torch
from torch import nn


def pytorch_dense_to_jax_params(model: nn.Module) -> Dict[str, Any]:
    """
    Convert a ftlelab.models.dense.DenseNet instance to a JAX params dict
    for use with dense_forward / dense_hidden_k.

    Assumes model.net is a Sequential of:
      Linear, activation, (Dropout), ..., Linear, activation
    and all Linear layers are the ones you want to mirror in order.
    """
    layers = []
    for m in model.net:
        if isinstance(m, nn.Linear):
            W = m.weight.detach().cpu().numpy()  # [out_dim, in_dim]
            b = m.bias.detach().cpu().numpy()    # [out_dim]
            layers.append(
                {
                    "W": jnp.asarray(W),
                    "b": jnp.asarray(b),
                }
            )
    return {"layers": layers}