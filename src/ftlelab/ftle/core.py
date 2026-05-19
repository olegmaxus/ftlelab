from dataclasses import dataclass
from typing import Tuple, Union
import math
import torch
import torch.nn as nn

# Prefer torch.func.jvp; fallback to functorch; else finite differences
_HAS_TORCH_FUNC = hasattr(torch, "func") and hasattr(torch.func, "jvp")
try:
    from functorch import jvp as ft_jvp
    _HAS_FUNCTORCH = True
except Exception:
    _HAS_FUNCTORCH = False

LayerSpec = Union[str, Tuple[str, int]]  # "output" or ("hidden_k", K)
_ACTS = (nn.Tanh, nn.ReLU, nn.GELU, nn.Sigmoid, nn.LeakyReLU, nn.Softplus)

@dataclass
class SVConfig:
    iters: int = 20             # PI iterations
    tol: float = 1e-6           # PI convergence tolerance on ||J v||^2
    fd_eps: float = 1e-4        # finite-diff step for JVP fallback
    exact_if_dim_le: int = 4    # if d <= this, use exact SVD of J
    jvp_backend: str = "auto"   # "auto"|"torch"|"functorch"|"fd"

def _device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device

def _ensure_vec1d(y: torch.Tensor) -> torch.Tensor:
    if y.ndim == 2 and y.size(0) == 1:  # [1, m] -> [m]
        return y.squeeze(0)
    return y

def _dot(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a * b).sum().item()

# ---------- Build f_K(x) ----------
def _f_output(model: nn.Module):
    def f(x: torch.Tensor) -> torch.Tensor:
        y = model(x)
        return y.reshape(()) if y.numel() == 1 else _ensure_vec1d(y)
    return f

def _f_hidden_k_via_method(model: nn.Module, K: int):
    def f(x: torch.Tensor) -> torch.Tensor:
        return _ensure_vec1d(model.hidden_k(x, K))  # post-activation helper
    return f

def _f_hidden_k_via_hooks(model: nn.Module, K: int):
    def f(x: torch.Tensor) -> torch.Tensor:
        cap = {"t": None}; cnt = {"n": 0}
        def hook(_m, _inp, out):
            if isinstance(_m, _ACTS):
                cnt["n"] += 1
                if cnt["n"] == K: cap["t"] = out
        hs = []
        for m in model.modules():
            if isinstance(m, _ACTS):
                hs.append(m.register_forward_hook(hook))
        try:
            _ = model(x)
        finally:
            for h in hs: h.remove()
        if cap["t"] is None:
            raise RuntimeError(f"Could not capture hidden_k={K}")
        return _ensure_vec1d(cap["t"])
    return f

def build_feature_fn(model: nn.Module, layer_spec: LayerSpec):
    model.eval()
    if layer_spec == "output":
        return _f_output(model)
    if isinstance(layer_spec, tuple) and layer_spec[0] == "hidden_k":
        K = int(layer_spec[1])
        return _f_hidden_k_via_method(model, K) if hasattr(model, "hidden_k") else _f_hidden_k_via_hooks(model, K)
    raise ValueError("layer_spec must be 'output' or ('hidden_k', K).")

# ---------- JVP/VJP utilities ----------
def _match_tangent_shape(x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Make v the same shape as x for jvp."""
    if v.shape == x.shape:
        return v
    if x.ndim == 2 and x.size(0) == 1 and v.ndim == 1 and v.size(0) == x.size(1):
        return v.unsqueeze(0)
    # generic fallback
    return v.reshape_as(x)

def _jvp(fun, x, v, backend="auto", fd_eps=1e-4):
    v = _match_tangent_shape(x, v).to(x.dtype)
    if backend in ("auto", "torch"):
        if _HAS_TORCH_FUNC:
            y, jv = torch.func.jvp(fun, (x,), (v,))
            return jv
        if backend == "torch":
            raise RuntimeError("Requested torch.func.jvp but it's unavailable.")
    if backend in ("auto", "functorch"):
        if _HAS_FUNCTORCH:
            y, jv = ft_jvp(fun, (x,), (v,))
            return jv
        if backend == "functorch":
            raise RuntimeError("Requested functorch.jvp but it's unavailable.")
    # finite-diff fallback
    return (fun(x + fd_eps * v) - fun(x)) / fd_eps

def _jtj_mv(fun, x, v, backend="auto", fd_eps=1e-4):
    """
    Compute (J^T J) v using JVP (J v) and VJP (J^T u), without forming J.
    x: [1, d] (or [d]), v: [d]; returns [d].
    """
    x = x.detach().requires_grad_(True)
    y = fun(x)
    if y.ndim == 0:
        raise RuntimeError("Scalar feature in _jtj_mv; use grad-norm path.")
    jv = _jvp(fun, x, v, backend=backend, fd_eps=fd_eps)                  # [m]
    Av = torch.autograd.grad(y, x, grad_outputs=jv, retain_graph=True)[0] # [1, d] or [d]
    return Av.squeeze(0)

def _orthonormalize_columns(X: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    if X.ndim == 1:
        X = X.unsqueeze(1)
    Q, R = torch.linalg.qr(X, mode="reduced")
    return Q

def _normalize(v: torch.Tensor, eps=1e-12):
    n = v.norm()
    return v / (n + eps), n

# ---------- Exact Jacobian + SVD (small d) ----------
def _jacobian_columns_by_jvp(fun, x, d, backend="auto", fd_eps=1e-4):
    """
    Build J ∈ R^{m x d} column-wise: J e_i = JVP(fun, x, e_i) with e_i shaped like x.
    """
    if backend in ("auto", "torch") and _HAS_TORCH_FUNC and hasattr(torch.func, "vmap"):
        x_flat = x.reshape(-1)
        I = torch.eye(d, device=x.device, dtype=x.dtype)

        def single_jvp(v_flat):
            v = v_flat.view_as(x)
            jv = _jvp(fun, x, v, backend=backend, fd_eps=fd_eps)
            if jv.ndim == 0:
                jv = jv.unsqueeze(0)
            return jv.reshape(-1)

        J_cols = torch.func.vmap(single_jvp)(I) # [d, m]
        return J_cols.transpose(0, 1).contiguous() # [m, d]
    
    cols = []
    for i in range(d):
        e = torch.zeros_like(x).view(-1)
        e[i] = 1.0
        e = e.view_as(x)
        jv = _jvp(fun, x, e, backend=backend, fd_eps=fd_eps)
        if jv.ndim == 0:
            jv = jv.unsqueeze(0)
        cols.append(jv.detach().reshape(-1))
    J = torch.stack(cols, dim=1)
    return J

def exact_svals_and_V(fun, x, backend="auto", fd_eps=1e-4):
    """
    Return (S, V) where S are singular values (descending) and V has right singular
    vectors as columns. For thin SVD we get V ∈ R^{d x k} with k = min(m, d).
    """
    x = x.detach().requires_grad_(True)
    y = fun(x)
    d = x.numel()

    # Scalar feature: J is [1×d], so ||grad|| is σ1 and v1 ∝ grad
    if y.ndim == 0:
        g = torch.autograd.grad(y, x, retain_graph=False, create_graph=False)[0].reshape(-1)
        s = g.norm()
        v1 = g / (s + 1e-12)
        return torch.tensor([s], device=x.device), v1.unsqueeze(1)  # S:[1], V:[d,1]

    # Build J column-wise via JVP
    J = _jacobian_columns_by_jvp(fun, x, d, backend=backend, fd_eps=fd_eps)  # [m, d]

    # Thin SVD (preferred): U:[m,k], S:[k], Vh:[k,d], k = min(m,d)
    U, S, Vh = torch.linalg.svd(J, full_matrices=False)
    V = Vh.transpose(-2, -1)  # V:[d, k]
    return S, V

# ---------- Top-1 and Top-2 ----------
def _top1_sigma_from_fn(
    fK: callable,
    x: torch.Tensor,
    cfg: SVConfig = SVConfig(),
):
    """
    Core of top1_sigma: operates on an already-built feature function fK.
    """
    xr = x.detach().requires_grad_(True)
    y0 = fK(xr)

    # Scalar shortcut: sigma_1 = \|\grad f(x)\|
    if y0.ndim == 0:
        g = torch.autograd.grad(y0, xr, retain_graph=False, create_graph=False)[0].reshape(-1)
        s1 = float(g.norm().item())
        v1 = (g / (g.norm() + 1e-12)).detach()
        return s1, v1
    
    d_total = x.detach().numel()
    if d_total <= cfg.exact_if_dim_le: # if dimension is small enough, just do exact SVD
        S, V = exact_svals_and_V(fK, xr, backend=cfg.jvp_backend, fd_eps=cfg.fd_eps)
        return float(S[0].item()), V[:, 0].detach()


    # Power iteration on J^\top J
    v, _ = _normalize(torch.randn_like(xr)) # device and dtype of xr are captured automatically
    last = None
    for _ in range(cfg.iters):
        Av = _jtj_mv(fK, xr, v, backend=cfg.jvp_backend, fd_eps=cfg.fd_eps)
        v, _ = _normalize(Av)
        jv = _jvp(fK, xr, v, backend=cfg.jvp_backend, fd_eps=cfg.fd_eps)
        rq2 = float(jv.norm().item() ** 2)
        if last is not None and abs(rq2 - last) < cfg.tol * max(1.0, abs(last)):
            break
        last = rq2
    s1 = math.sqrt(max(last or 0.0, 0.0))
    return s1, v.detach()


def top1_sigma(
    model: nn.Module,
    x: torch.Tensor,
    layer_spec: LayerSpec,
    cfg: SVConfig = SVConfig(),
) -> Tuple[float, torch.Tensor]:
    """
    Return (sigma_1, v_1) for J_K at input x.
    """
    device = _device(model)
    x = x.to(device)

    if x.ndim in (1, 3):
        x = x.unsqueeze(0)  # [1, d] / [1, C, H, W]

    fK = build_feature_fn(model, layer_spec)
    return _top1_sigma_from_fn(fK, x, cfg)


def top1_sigma_batch(
    model: nn.Module,
    X: torch.Tensor,           # [B, d] or [B, ...]
    layer_spec: LayerSpec,
    cfg: SVConfig = SVConfig()
) -> torch.Tensor:
    """
    Approximate sigma_1(J_K(x)) for each x in X.
    Returns: s1 of shape [B], values only (no vectors).
    Currently just loops over batch in Python, but centralized.
    """
    device = _device(model)
    X = X.to(device)

    if X.ndim == 1:
        X = X.unsqueeze(0)  # [1, d]

    fK = build_feature_fn(model, layer_spec)

    sigmas = []
    for x in X:
        if x.ndim in (1, 3):
            x = x.unsqueeze(0)
        s1, _ = _top1_sigma_from_fn(fK, x, cfg)
        sigmas.append(s1)

    return torch.tensor(sigmas, dtype=torch.float32, device=device)

def _top2_sigmas_from_fn(
    fK: callable,
    x: torch.Tensor,
    cfg: SVConfig = SVConfig(),
) -> Tuple[float, torch.Tensor, float, torch.Tensor]:
    """
    Core of top2_sigmas: operates on an already-built feature function fK.
    """
    xr = x.detach().requires_grad_(True)
    y0 = fK(xr)

    if y0.ndim == 0:
        g = torch.autograd.grad(y0, xr, retain_graph=False)[0].reshape(-1)
        s1 = float(g.norm().item())
        v1 = (g / (g.norm() + 1e-12)).detach()
        return s1, v1, 0.0, torch.zeros_like(v1)

    d_total = x.detach().numel()

    # Exact path for small d
    if d_total <= cfg.exact_if_dim_le:
        S, V = exact_svals_and_V(fK, xr, backend=cfg.jvp_backend, fd_eps=cfg.fd_eps)
        s1 = float(S[0].item()); v1 = V[:, 0].detach()
        s2 = float(S[1].item()) if S.numel() > 1 else 0.0
        v2 = V[:, 1].detach() if V.size(1) > 1 else torch.zeros_like(v1)
        return s1, v1, s2, v2

    d = d_total
    V = torch.randn(d, 2, device=xr.device, dtype=xr.dtype)
    V = _orthonormalize_columns(V)

    last = None
    for _ in range(cfg.iters):
        cols = [
            _jtj_mv(fK, xr, V[:, j].view_as(xr),
                    backend=cfg.jvp_backend,
                    fd_eps=cfg.fd_eps).reshape(-1)
            for j in range(2)
        ]

        Z = torch.stack(cols, dim=1)
        V_new = _orthonormalize_columns(Z)
        B = V_new.t().mm(Z) # V^T (J^T J) V
        eigvals_approx = torch.linalg.eigvalsh(B)
        lambda_1 = float(eigvals_approx[-1].item()) # largest eigenvalue

        if last is not None and abs(lambda_1 - last) < cfg.tol * max(1.0, abs(last)):
            V = V_new
            break
        V = V_new
        last = lambda_1

    cols = [
        _jtj_mv(fK, xr, V[:, j].view_as(xr),
                backend=cfg.jvp_backend,
                fd_eps=cfg.fd_eps).reshape(-1)
        for j in range(2)
    ]

    Z = torch.stack(cols, dim=1)
    B = V.t().mm(Z)
    eigvals, idx = torch.linalg.eigvalsh(B).sort(descending=True)
    sigmas = torch.sqrt(torch.clamp(eigvals, min=0.0))
    V = V[:, idx]

    return (
        float(sigmas[0].item()), V[:, 0].detach(),
        float(sigmas[1].item()), V[:, 1].detach(),
    )


def top2_sigmas(model: nn.Module,
                x: torch.Tensor,
                layer_spec: LayerSpec,
                cfg: SVConfig = SVConfig()) -> Tuple[float, torch.Tensor, float, torch.Tensor]:
    """
    Return (sigma_1, v1, sigma_2, v2). If rank<=1, sigma_2 approx 0 and v2 is zeros.
    """
    device = _device(model)
    x = x.to(device)

    if x.ndim in (1, 3): 
        x = x.unsqueeze(0)

    fK = build_feature_fn(model, layer_spec)

    return _top2_sigmas_from_fn(fK, x, cfg)

def top2_sigma_batch(
    model: nn.Module,
    X: torch.Tensor,
    layer_spec: LayerSpec,
    cfg: SVConfig = SVConfig()
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return sigma_1, sigma_2 for each x in X as two [B] tensors.
    """
    device = _device(model)
    X = X.to(device)

    if X.ndim == 1:
        X = X.unsqueeze(0)

    fK = build_feature_fn(model, layer_spec)

    sigmas1, sigmas2 = [], []
    for x in X:
        if x.ndim in (1, 3):
            x = x.unsqueeze(0)
        s1, _, s2, _ = _top2_sigmas_from_fn(fK, x, cfg)
        sigmas1.append(s1)
        sigmas2.append(s2)

    return (
        torch.tensor(sigmas1, dtype=torch.float32, device=device),
        torch.tensor(sigmas2, dtype=torch.float32, device=device),
    )

def topk_sigmas(
    model: nn.Module,
    x: torch.Tensor,
    layer_spec: LayerSpec,
    k: int = 1,
    cfg: SVConfig = SVConfig()
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Approximate top-k singular values sigma_1, ..., sigma_k and corresponding right singular
    vectors v_1, ..., v_k of J_K(x) via block power iteration on A = J^T J.

    Returns:
      sigmas: [k] tensor (descending)
      V:      [d, k] tensor, columns ~ v_1..v_k
    """
    assert k >= 1, "k must be >= 1"

    device = _device(model)
    x = x.to(device)

    if x.ndim in (1, 3):
        x = x.unsqueeze(0)  # [1, d] / [1, C, H, W]

    fK = build_feature_fn(model, layer_spec)

    xr = x.detach().requires_grad_(True)
    y0 = fK(xr)

    # Handle scalar feature: rank-1 Jacobian
    if y0.ndim == 0:
        g = torch.autograd.grad(y0, xr, retain_graph=False, create_graph=False)[0].reshape(-1)
        s = g.norm()
        v1 = g / (s + 1e-12)
        sigmas = torch.zeros(k, device=device)
        sigmas[0] = s
        V = torch.zeros(g.numel(), k, device=device)
        V[:, 0] = v1
        return sigmas, V

    d_total = xr.detach().numel()

    # Exact small-d path: build full J and SVD
    if d_total <= cfg.exact_if_dim_le:
        S, V_full = exact_svals_and_V(fK, xr, backend=cfg.jvp_backend, fd_eps=cfg.fd_eps)
        k_eff = min(k, S.numel(), V_full.size(1))
        sigmas = S[:k_eff].clone()
        V = V_full[:, :k_eff].detach()
        # If k > k_eff, pad with zeros
        if k_eff < k:
            pad_s = torch.zeros(k - k_eff, device=device)
            pad_V = torch.zeros(V.size(0), k - k_eff, device=device)
            sigmas = torch.cat([sigmas, pad_s], dim=0)
            V = torch.cat([V, pad_V], dim=1)
        return sigmas, V

    # Block power iteration on A = J^T J
    d = d_total
    # Initialize random subspace V: [d, k]
    V = torch.randn(d, k, device=device, dtype=xr.dtype)
    V = _orthonormalize_columns(V)

    last = None
    for _ in range(cfg.iters):
        # Apply A = J^T J to each column of V
        cols = []
        for j in range(k):
            vj = V[:, j].view_as(xr)          # shape like x
            Avj = _jtj_mv(fK, xr, vj, backend=cfg.jvp_backend, fd_eps=cfg.fd_eps)
            cols.append(Avj.reshape(-1))      # flatten
        Z = torch.stack(cols, dim=1)          # [d, k]

        # Orthonormalize
        V_new = _orthonormalize_columns(Z)

        # Optional convergence check: use Rayleigh quotient trace
        # B = V^T (A V) ≈ V^T Z (k×k). Its diag entries ≈ eigenvalues.
        B = V_new.t().mm(Z)                   # [k, k]
        eigvals_approx = torch.linalg.eigvalsh(B)  # [k]
        eigvals_approx, _ = torch.sort(eigvals_approx, descending=True)
        # Take largest eigenvalue as scalar to check convergence
        lambda_1 = float(eigvals_approx[0].item())

        if last is not None:
            if abs(lambda_1 - last) < cfg.tol * max(1.0, last):
                V = V_new
                break
        V = V_new
        last = lambda_1

    # Final Rayleigh matrix B = V^T (A V) for better eigenvalue estimates
    cols = []
    for j in range(k):
        vj = V[:, j].view_as(xr)
        Avj = _jtj_mv(fK, xr, vj, backend=cfg.jvp_backend, fd_eps=cfg.fd_eps)
        cols.append(Avj.reshape(-1))
    Z = torch.stack(cols, dim=1)  # [d, k]
    B = V.t().mm(Z)               # [k, k]

    # Eigenvalues of B ≈ eigenvalues of J^T J
    eigvals = torch.linalg.eigvalsh(B)
    eigvals, idx = torch.sort(eigvals, descending=True)

    sigmas = torch.sqrt(torch.clamp(eigvals, min=0.0))
    # Sort V accordingly (columns ordered by eigenvalues)
    V = V[:, idx]

    return sigmas, V