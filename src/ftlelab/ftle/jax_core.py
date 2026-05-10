from typing import Any, Callable, Dict, Tuple, Union

import numpy as np
import jax
import jax.numpy as jnp
from tqdm.auto import tqdm


from .jax_models import dense_forward, dense_hidden_k


# ============================================================
# Types
# ============================================================

LayerSpec = Union[str, Tuple[str, int]]  # "output" or ("hidden_k", K)


# ============================================================
# Feature function builder (JAX)
# ============================================================


def build_feature_fn_jax(
    model_type: str,
    params: Dict[str, Any],
    layer_spec: LayerSpec,
    *,
    activation: str = "tanh",
    output_activation: str = "tanh",
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """
    model_type: "dense" | "conv" | "autoencoder" | "vae" | ...
    layer_spec: "output" or ("hidden_k", k) or ("latent",) etc.

    Returns:
      f(x): jnp.ndarray feature vector corresponding to the requested layer.
    """
    if model_type == "dense":
        if layer_spec == "output":
            return lambda x: dense_forward(
                params, x, activation=activation, output_activation=output_activation
            )
        if isinstance(layer_spec, tuple) and layer_spec[0] == "hidden_k":
            k = int(layer_spec[1])
            return lambda x: dense_hidden_k(params, x, k, activation=activation)

        raise ValueError(
            "For model_type='dense', layer_spec must be 'output' or ('hidden_k', k)."
        )

    # Placeholders for future extensions:
    elif model_type == "conv":
        ...
    elif model_type == "autoencoder":
        ...
    elif model_type == "vae":
        ...

    raise ValueError(f"Unknown model_type: {model_type}")


# ============================================================
# JVP/VJP helper: (J^T J) v
# ============================================================


def jtj_mv(
    f: Callable[[jnp.ndarray], jnp.ndarray],
    x: jnp.ndarray,
    v: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute (J^T J) v using JAX JVP and VJP, without forming J.

    f: R^d -> R^m
    x: (d,)
    v: (d,)
    return: (d,)
    """

    dtype = x.dtype
    x = x.astype(dtype)
    v = v.astype(dtype)

    # J v
    _, Jv = jax.jvp(f, (x,), (v,))
    # J^T (J v)
    _, vjp_fun = jax.vjp(f, x)
    JtJv, = vjp_fun(Jv)
    return JtJv


def orthonormalize(V: jnp.ndarray) -> jnp.ndarray:
    """
    Orthonormalize columns of V via QR.

    V: (d, k)
    Returns: Q: (d, k)
    """
    Q, _ = jnp.linalg.qr(V)
    return Q


# ============================================================
# Top-k singular values (values only) via block power iteration
# ============================================================


# def topk_singular_values_jax(
#     f: Callable[[jnp.ndarray], jnp.ndarray],
#     x: jnp.ndarray,
#     k: int = 1,
#     n_steps: int = 20,
#     key: jax.Array | None = None,
# ) -> jnp.ndarray:
#     """
#     Approximate top-k singular values σ_1..σ_k of J_f(x) via block power
#     iteration on A = J^T J.

#     f: R^d -> R^m
#     x: (d,)
#     k: number of singular values to approximate
#     n_steps: number of power iterations
#     returns: sigmas: (k,) sorted in descending order
#     """
#     d = x.shape[0]
#     if key is None:
#         key = jax.random.PRNGKey(0)

#     # Random initial subspace
#     V = jax.random.normal(key, (d, k))
#     V = orthonormalize(V)

#     def body(V, _):
#         # Apply A = J^T J to each column of V
#         Z_cols = [jtj_mv(f, x, V[:, j]) for j in range(k)]
#         Z = jnp.stack(Z_cols, axis=1)  # (d, k)
#         V_new = orthonormalize(Z)
#         return V_new, None

#     V_final, _ = jax.lax.scan(body, V, None, length=n_steps)

#     # Form Rayleigh matrix B = V^T (A V)
#     Z_cols = [jtj_mv(f, x, V_final[:, j]) for j in range(k)]
#     Z = jnp.stack(Z_cols, axis=1)         # (d, k)
#     B = V_final.T @ Z                     # (k, k)

#     eigvals = jnp.linalg.eigvalsh(B)      # ascending
#     eigvals = jnp.sort(eigvals)[::-1]     # descending
#     sigmas = jnp.sqrt(jnp.clip(eigvals, a_min=0.0))
#     return sigmas

def topk_singular_values_jax(
    f: Callable[[jnp.ndarray], jnp.ndarray],
    x: jnp.ndarray,
    k: int = 1,
    max_steps: int = 50,
    tol: float = 1e-6,
    key: jax.Array | None = None,
) -> jnp.ndarray:
    """
    Approximate top-k singular values σ_1..σ_k of J_f(x) via block power
    iteration on A = J^T J, with a convergence check based on the leading
    eigenvalue estimate (≈ ||J v||^2).

    f: R^d -> R^m
    x: (d,)
    k: number of singular values to approximate
    max_steps: maximum number of power-iteration steps
    tol: relative tolerance on change of leading eigenvalue
    returns: sigmas: (k,) sorted in descending order
    """
    d = x.shape[0]
    if key is None:
        key = jax.random.PRNGKey(0)

    # Random initial subspace
    V0 = jax.random.normal(key, (d, k))
    V0 = orthonormalize(V0)

    def step(carry, _):
        V, last_lambda, done = carry

        def do_iter(args):
            V, last_lambda = args

            # Apply A = J^T J to each column of V
            Z_cols = [jtj_mv(f, x, V[:, j]) for j in range(k)]
            Z = jnp.stack(Z_cols, axis=1)  # (d, k)

            # Orthonormalize to get new subspace
            V_new = orthonormalize(Z)

            # Rayleigh matrix B = V^T (A V) = V^T Z
            B = V_new.T @ Z  # (k, k)
            eigvals = jnp.linalg.eigvalsh(B)
            eigvals = jnp.sort(eigvals)[::-1]  # descending
            lambda1 = eigvals[0]

            # Relative change in leading eigenvalue
            # use max(1, |last_lambda|) for scaling
            scale = jnp.maximum(1.0, jnp.abs(last_lambda))
            delta = jnp.abs(lambda1 - last_lambda) / scale

            new_done = delta < tol

            return (V_new, lambda1, new_done), (lambda1, delta)

        def keep(args):
            return (V, last_lambda, done), (last_lambda, jnp.nan)

        # Only update if not done yet
        (V_next, lam_next, done_next), info = jax.lax.cond(
            done, keep, do_iter, (V, last_lambda)
        )
        return (V_next, lam_next, done_next), info

    # Run max_steps iterations, but allow early convergence
    # Initialize last_lambda = 0.0, done = False
    init_carry = (V0, jnp.array(0.0, dtype=x.dtype), jnp.array(False))
    (V_final, lambda_final, done_final), infos = jax.lax.scan(
        step, init_carry, None, length=max_steps
    )

    # After the loop, V_final is the last subspace (possibly converged early)
    # Recompute Z and B for final eigenvalue estimates
    Z_cols = [jtj_mv(f, x, V_final[:, j]) for j in range(k)]
    Z = jnp.stack(Z_cols, axis=1)           # (d, k)
    B = V_final.T @ Z                       # (k, k)

    eigvals = jnp.linalg.eigvalsh(B)
    eigvals = jnp.sort(eigvals)[::-1]
    sigmas = jnp.sqrt(jnp.clip(eigvals, a_min=0.0))
    return sigmas

# ============================================================
# FTLE (maximal, k=1) at a single point
# ============================================================


def ftle_at_point(
    model_type: str,
    params: Dict[str, Any],
    x: jnp.ndarray,
    layer_spec: LayerSpec,
    time_L: int,
    *,
    activation: str = "tanh",
    output_activation: str = "tanh",
    max_steps: int = 50,
) -> jnp.ndarray:
    """
    Compute maximal FTLE λ_1(x) = (1/L) log σ_1(J_f(x)) for the layer
    specified by (model_type, layer_spec).

    model_type: "dense" | "conv" | ...
    params: JAX params dict matching the chosen architecture
    x: (d,) input point
    layer_spec: e.g. "output" or ("hidden_k", K)
    time_L: "time" horizon (e.g. number of hidden layers up to this layer)
    """
    f = build_feature_fn_jax(
        model_type,
        params,
        layer_spec,
        activation=activation,
        output_activation=output_activation,
    )

    sigmas = topk_singular_values_jax(f, x, k=1, max_steps=max_steps)
    sigma1 = sigmas[0]
    lam = (1.0 / max(int(time_L), 1)) * jnp.log(jnp.maximum(sigma1, 1e-12))
    return lam


# ============================================================
# FTLE field over many points (vectorized + JIT)
# ============================================================


def ftle_field(
    model_type: str,
    params: Dict[str, Any],
    X: jnp.ndarray,           # (N, d)
    layer_spec: LayerSpec,
    time_L: int,
    *,
    activation: str = "tanh",
    output_activation: str = "tanh",
    max_steps: int = 50,
) -> jnp.ndarray:
    """
    Compute maximal FTLE λ_1(x) at all x in X.

    Returns:
      lam: (N,) array of FTLE values
    """
    def ftle_single(x):
        return ftle_at_point(
            model_type,
            params,
            x,
            layer_spec,
            time_L,
            activation=activation,
            output_activation=output_activation,
            max_steps=max_steps,
        )

    ftle_vmapped = jax.jit(jax.vmap(ftle_single))
    return ftle_vmapped(X)


def ftle_field_batched(
    model_type,
    params,
    X_np: np.ndarray,     # (N, d), numpy
    layer_spec,
    time_L: int,
    batch_size: int = 1024,
    activation: str = "tanh",
    output_activation: str = "tanh",
    max_steps: int = 50,
    tol: float = 1e-6,
    dtype: str = "float64", # float32 or float64
) -> np.ndarray:
    """
    Compute FTLE over X_np in batches, showing tqdm progress.
    Returns FTLE values as a numpy array of shape (N,).
    """

    if dtype == "float64":
        jax.config.update("jax_enable_x64", True)
        jax_dtype = jnp.float64
    elif dtype == "float32":
        jax.config.update("jax_enable_x64", False)
        jax_dtype = jnp.float32
    else:
        raise ValueError("Unsupported dtype: choose 'float32' or 'float64'.")
    
    # Single-point FTLE (pure JAX)
    def ftle_single(x):
        x = x.astype(jax_dtype)
        return ftle_at_point(
            model_type=model_type,
            params=params,
            x=x,
            layer_spec=layer_spec,
            time_L=time_L,
            activation=activation,
            output_activation=output_activation,
            max_steps=max_steps,
        )

    # Batched version
    ftle_batch = jax.jit(jax.vmap(ftle_single))

    N = X_np.shape[0]
    out = np.empty((N,), dtype=np.float32)

    for start in tqdm(range(0, N, batch_size), desc="FTLE (JAX)"):
        end = min(start + batch_size, N)
        X_batch = jnp.asarray(X_np[start:end])      # (B, d)
        lam_batch = np.array(ftle_batch(X_batch))  # back to numpy
        out[start:end] = lam_batch.astype(np.float32)

    return out