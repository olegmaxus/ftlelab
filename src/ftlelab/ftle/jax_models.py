import jax
import jax.numpy as jnp
from typing import Any, Dict, Sequence


# ============================================================
# Dense network (JAX) mirroring ftlelab.models.dense.DenseNet
# ============================================================


def dense_forward(
    params: Dict[str, Any],
    x: jnp.ndarray,
    activation: str = "tanh",
    output_activation: str = "tanh",
) -> jnp.ndarray:
    """
    Forward pass for a fully-connected network with arbitrary layer widths.

    params["layers"] is a list of dicts:
        {"W": [out_dim, in_dim], "b": [out_dim]}

    x: (..., in_dim)
    """
    acts = {
        "tanh": jnp.tanh,
        "relu": jax.nn.relu,
        "gelu": jax.nn.gelu,
        "leaky_relu": jax.nn.leaky_relu,
        "softplus": jax.nn.softplus,
        "sigmoid": jax.nn.sigmoid,
        "identity": lambda z: z,
    }
    act = acts[activation]
    out_act = acts.get(output_activation, acts["identity"])

    h = x
    L = len(params["layers"])
    for i, layer in enumerate(params["layers"]):
        W, b = layer["W"], layer["b"]
        h = h @ W.T + b
        if i < L - 1:
            h = act(h)
        else:
            h = out_act(h)
    return h


def dense_hidden_k(
    params: Dict[str, Any],
    x: jnp.ndarray,
    k: int,
    activation: str = "tanh",
) -> jnp.ndarray:
    """
    Post-activation output of the k-th hidden layer (1-based) for DenseNet.

    Layer dims are [in_dim, h1, ..., hL, out_dim]; hidden layers are 1..L.
    """
    acts = {
        "tanh": jnp.tanh,
        "relu": jax.nn.relu,
        "gelu": jax.nn.gelu,
        "leaky_relu": jax.nn.leaky_relu,
        "softplus": jax.nn.softplus,
        "sigmoid": jax.nn.sigmoid,
        "identity": lambda z: z,
    }
    act = acts[activation]

    h = x
    # all but last Linear are hidden
    for i, layer in enumerate(params["layers"][:-1], start=1):
        W, b = layer["W"], layer["b"]
        h = h @ W.T + b
        h = act(h)
        if i == k:
            return h
    raise ValueError(f"k={k} exceeds hidden depth {len(params['layers']) - 1}")