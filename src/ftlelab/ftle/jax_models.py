import jax
import jax.numpy as jnp
from typing import Any, Dict, Sequence, Union, Tuple

LayerSpec = Union[str, Tuple[str, int]]  # "output" or ("hidden_k", K)

def _get_acts():
    return {
        "tanh": jnp.tanh,
        "relu": jax.nn.relu,
        "gelu": jax.nn.gelu,
        "leaky_relu": jax.nn.leaky_relu,
        "softplus": jax.nn.softplus,
        "identity": lambda z: z,
    }

# ============================================================
# Dense network (JAX) mirroring ftlelab.models.dense.DenseNet
# ============================================================

def _layer_depth_index(
    params: Dict[str, Any],
    layer_spec: LayerSpec
) -> int:
    hidden_depth = len(params["layers"]) - 1
    if layer_spec == "input":
        return 0
    if layer_spec == "output":
        return hidden_depth + 1
    if isinstance(layer_spec, tuple) and layer_spec[0] == "hidden_k":
        k = int(layer_spec[1])
        if not (1 <= k <= hidden_depth):
            raise ValueError(f"hidden_k must be in [1, {hidden_depth}], got {k}")
        return k
    raise ValueError(f"Unsupported layer_spec: {layer_spec}")


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
    acts = _get_acts()
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
    acts = _get_acts()
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


def dense_map_between(
    params: Dict[str, Any],
    z: jnp.ndarray,
    start_layer_spec: LayerSpec,
    end_layer_spec: LayerSpec,
    activation: str = "tanh",
    output_activation: str = "tanh",      
) -> jnp.ndarray:
    """
    Map a representation at start_layer_spec to the representation at end_layer_spec.

    start_layer_spec: "input" or ("hidden_k", s)
    end_layer_spec:   ("hidden_k", t) or "output"
    """
    acts = _get_acts()
    act = acts[activation]
    out_act = acts.get(output_activation, acts["identity"])

    start_idx = _layer_depth_index(params, start_layer_spec)
    end_idx = _layer_depth_index(params, end_layer_spec)
    n_linear = len(params["layers"])

    if not (0 <= start_idx < end_idx <= n_linear):
        raise ValueError(
            f"Need start before end, got {start_layer_spec} -> {end_layer_spec}"
        )
    
    h = z
    for linear_idx in range(start_idx, end_idx):
        W, b = params["layers"][linear_idx]["W"], params["layers"][linear_idx]["b"]
        h = h @ W.T + b
        if linear_idx == n_linear - 1:
            h = out_act(h)
        else:
            h = act(h)
    return h


# ============================================================
# Autoencoder (JAX) mirroring ftlelab.models.autoencoder.AutoEncoder
# ============================================================

def _ae_layer_depth_index(
    params: Dict[str, Any],
    layer_spec: LayerSpec,
) -> int:
    enc_h = int(params["encoder_hidden_depth"])
    dec_h = int(params["decoder_hidden_depth"])

    if layer_spec == "input":
        return 0
    if layer_spec == "latent":
        return enc_h + 1
    if layer_spec == "output":
        return enc_h + 1 + dec_h + 1
    if isinstance(layer_spec, tuple):
        name, k = layer_spec[0], int(layer_spec[1])
        if name == "encoder_hidden_k":
            if not (1 <= k <= enc_h):
                raise ValueError(f"encoder_hidden_k must be in [1, {enc_h}], got {k}")
            return k
        if name == "decoder_hidden_k":
            if not (1 <= k <= dec_h):
                raise ValueError(f"decoder_hidden_k must be in [1, {dec_h}], got {k}")
            return enc_h + 1 + k
    raise ValueError(f"Unsupported autoencoder layer_spec: {layer_spec}")


def ae_map_between(
    params: Dict[str, Any],
    h: jnp.ndarray,
    start_layer_spec: LayerSpec,
    end_layer_spec: LayerSpec,
    activation: str = "relu",
    output_activation: str = "identity",
) -> jnp.ndarray:
    acts = _get_acts()
    act = acts[activation]
    out_act = acts.get(output_activation, acts["identity"])

    start_idx = _ae_layer_depth_index(params, start_layer_spec)
    end_idx = _ae_layer_depth_index(params, end_layer_spec)
    if not (0 <= start_idx < end_idx):
        raise ValueError(
            f"Need start before end, got {start_layer_spec} -> {end_layer_spec}"
        )

    enc = params["encoder_layers"]
    dec = params["decoder_layers"]
    enc_h = int(params["encoder_hidden_depth"])

    for idx in range(start_idx, end_idx):
        if idx < enc_h:
            layer = enc[idx]
            h = h @ layer["W"].T + layer["b"]
            h = act(h)
        elif idx == enc_h:
            layer = enc[enc_h]
            h = h @ layer["W"].T + layer["b"]
        elif idx < enc_h + int(params["decoder_hidden_depth"]) + 1:
            layer = dec[idx - enc_h - 1]
            h = h @ layer["W"].T + layer["b"]
            h = act(h)
        elif idx == enc_h + int(params["decoder_hidden_depth"]) + 1:
            layer = dec[-1]
            h = h @ layer["W"].T + layer["b"]
            h = out_act(h)
        else:
            raise ValueError(f"Invalid autoencoder transition index {idx}")

    return h


def ae_forward(
    params: Dict[str, Any],
    x: jnp.ndarray,
    activation: str = "relu",
    output_activation: str = "identity",
) -> jnp.ndarray:
    return ae_map_between(
        params,
        x,
        start_layer_spec="input",
        end_layer_spec="output",
        activation=activation,
        output_activation=output_activation,
    )


def layer_depth_index(
    model_type: str,
    params: Dict[str, Any],
    layer_spec: LayerSpec,
) -> int:
    if model_type == "dense":
        return _layer_depth_index(params, layer_spec)
    if model_type == "autoencoder":
        return _ae_layer_depth_index(params, layer_spec)
    raise ValueError(f"Unknown model_type: {model_type}")
