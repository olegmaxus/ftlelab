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

# ---------- Build f_K(x) ----------
def _f_output(model: nn.Module):
    def f(x: torch.Tensor) -> torch.Tensor:
        model.eval()
        return _ensure_vec1d(model(x))
    return f

def _f_hidden_k_via_method(model: nn.Module, K: int):
    def f(x: torch.Tensor) -> torch.Tensor:
        model.eval()
        return _ensure_vec1d(model.hidden_k(x, K))  # post-activation helper
    return f

def _f_hidden_k_via_hooks(model: nn.Module, K: int):
    def f(x: torch.Tensor) -> torch.Tensor:
        model.eval()
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

def _normalize(v: torch.Tensor, eps=1e-12):
    n = v.norm()
    return v / (n + eps), n

# ---------- Exact Jacobian + SVD (small d) ----------
def _jacobian_columns_by_jvp(fun, x, d, backend="auto", fd_eps=1e-4):
    """
    Build J ∈ R^{m x d} column-wise: J e_i = JVP(fun, x, e_i) with e_i shaped like x.
    """
    cols = []
    for i in range(d):
        e = torch.zeros_like(x)
        if x.ndim == 1:
            e[i] = 1.0
        else:  # [1, d]
            e[0, i] = 1.0
        jv = _jvp(fun, x, e, backend=backend, fd_eps=fd_eps)
        if jv.ndim == 0:
            jv = jv.unsqueeze(0)
        cols.append(jv.detach())
    J = torch.stack(cols, dim=1)  # [m, d]
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
def top1_sigma(model: nn.Module,
               x: torch.Tensor,
               layer_spec: LayerSpec,
               cfg: SVConfig = SVConfig()) -> Tuple[float, torch.Tensor]:
    """
    Return (σ1, v1) for J_K at input x.
    """
    device = _device(model)
    x = x.to(device)
    if x.ndim == 1: x = x.unsqueeze(0)  # [1, d]
    fK = build_feature_fn(model, layer_spec)

    xr = x.detach().requires_grad_(True)
    y0 = fK(xr)
    # Scalar shortcut: σ1 = ||∇ f(x)||
    if y0.ndim == 0:
        g = torch.autograd.grad(y0, xr, retain_graph=False, create_graph=False)[0].reshape(-1)
        s1 = float(g.norm().item())
        v1 = (g / (g.norm() + 1e-12)).detach()
        return s1, v1

    d = x.size(1)
    # Exact for small d
    if d <= cfg.exact_if_dim_le:
        S, V = exact_svals_and_V(fK, xr, backend=cfg.jvp_backend, fd_eps=cfg.fd_eps)
        return float(S[0].item()), V[:, 0].detach()

    # Power iteration on A = J^T J
    v = torch.randn(d, device=device); v, _ = _normalize(v)
    last = None
    for _ in range(cfg.iters):
        Av = _jtj_mv(fK, xr, v, backend=cfg.jvp_backend, fd_eps=cfg.fd_eps)
        v, _ = _normalize(Av)
        # Rayleigh via ||J v||^2
        jv = _jvp(fK, xr, v, backend=cfg.jvp_backend, fd_eps=cfg.fd_eps)
        rq2 = float(jv.norm().item() ** 2)
        if last is not None and abs(rq2 - last) < cfg.tol:
            break
        last = rq2
    s1 = math.sqrt(max(last or 0.0, 0.0))
    return s1, v.detach()

def top2_sigmas(model: nn.Module,
                x: torch.Tensor,
                layer_spec: LayerSpec,
                cfg: SVConfig = SVConfig()) -> Tuple[float, torch.Tensor, float, torch.Tensor]:
    """
    Return (sigma_1, v1, sigma_2, v2). If rank<=1, sigma_2 approx 0 and v2 is zeros.
    """
    device = _device(model)
    x = x.to(device)
    if x.ndim == 1: x = x.unsqueeze(0)
    fK = build_feature_fn(model, layer_spec)

    xr = x.detach().requires_grad_(True)
    y0 = fK(xr)

    # Scalar feature -> rank<=1
    if y0.ndim == 0:
        g = torch.autograd.grad(y0, xr, retain_graph=False, create_graph=False)[0].reshape(-1)
        s1 = float(g.norm().item())
        v1 = (g / (g.norm() + 1e-12)).detach()
        return s1, v1, 0.0, torch.zeros_like(v1)

    d = x.size(1)
    # Exact for small d
    if d <= cfg.exact_if_dim_le:
        S, V = exact_svals_and_V(fK, xr, backend=cfg.jvp_backend, fd_eps=cfg.fd_eps)
        s1 = float(S[0].item()); v1 = V[:, 0].detach()
        s2 = float(S[1].item()) if S.numel() > 1 else 0.0
        v2 = V[:, 1].detach() if V.size(1) > 1 else torch.zeros_like(v1)
        return s1, v1, s2, v2

    # Deflated PI for σ2
    s1, v1 = top1_sigma(model, xr, layer_spec, cfg)
    if s1 < 1e-12:
        return 0.0, v1, 0.0, torch.zeros_like(v1)

    v = torch.randn(d, device=device)
    v = v - (v @ v1) * v1        # orthogonalize to v1
    v, _ = _normalize(v)
    last = None

    for _ in range(cfg.iters):
        Av = _jtj_mv(fK, xr, v, backend=cfg.jvp_backend, fd_eps=cfg.fd_eps)
        Av_defl = Av - (s1**2) * (v1 @ v) * v1
        Av_defl = Av_defl - (Av_defl @ v1) * v1  # re-orthogonalize
        v, _ = _normalize(Av_defl)
        jv = _jvp(fK, xr, v, backend=cfg.jvp_backend, fd_eps=cfg.fd_eps)
        rq2 = float(jv.norm().item() ** 2)
        if last is not None and abs(rq2 - last) < cfg.tol:
            break
        last = rq2

    s2 = math.sqrt(max(last or 0.0, 0.0))
    return s1, v1.detach(), s2, v.detach()
