# ftle_grid.py
import math
import torch
import numpy as np
from typing import Tuple, Union, Optional
from tqdm.auto import tqdm

# Import the singular-value helpers you already have
from .core import SVConfig, top1_sigma, top2_sigmas, top1_sigma_batch, top2_sigma_batch, LayerSpec

def make_grid2d(
    x_min: float = -1.2, x_max: float = 1.2,
    y_min: float = -1.2, y_max: float = 1.2,
    nx: int = 200, ny: int = 200,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
):
    """
    Returns:
      grid:  [N, 2] tensor (row-major, x fastest), device=dtype as requested
      (XX, YY): meshgrid tensors shaped [ny, nx] on CPU for plotting
    """
    xs = torch.linspace(x_min, x_max, nx, dtype=dtype, device=device)
    ys = torch.linspace(y_min, y_max, ny, dtype=dtype, device=device)
    XX, YY = torch.meshgrid(xs, ys, indexing="xy")
    grid = torch.stack([XX.reshape(-1), YY.reshape(-1)], dim=1)
    # for plotting convenience return CPU mesh
    return grid, (XX.detach().cpu().reshape(ny, nx), YY.detach().cpu().reshape(ny, nx))

@torch.no_grad()
def grid_to_numpy(grid: torch.Tensor, nx: int, ny: int) -> Tuple[np.ndarray, np.ndarray]:
    """Reshape a flat [N] tensor into [ny, nx] numpy for plotting."""
    arr = grid.detach().cpu().numpy().reshape(ny, nx)
    return arr, arr  # (kept for API symmetry if you want two fields)

def compute_ftle_grid(
    model: torch.nn.Module,
    layer_spec: LayerSpec,     # "output" or ("hidden_k", K)
    time_L: int,               # divide log(sigma) by this "time" (layers up to K)
    grid: torch.Tensor,        # [N, 2] points (or [N, d] if your input is d-D)
    k: int = 1,                # 1 for σ1, 2 for σ2
    cfg: SVConfig = SVConfig(exact_if_dim_le=4, jvp_backend="auto"),
    batch_size: int = 1024,
    show_progress: bool = True,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Compute FTLE lambda_L(x) = (1/L) * log sigma_k(J_K(x)) over a grid.

    Returns:
      ftle: [N] tensor on CPU.
    """
    assert k in (1, 2), "k must be 1 or 2"
    model.eval()

    N = grid.size(0)
    vals = []
    rng = range(0, N, batch_size)
    if show_progress:
        rng = tqdm(rng,
                   total=(N + batch_size - 1) // batch_size,
                   desc=f"FTLE(k={k}) grid")

    # Work in model's device, but store results on CPU
    device = next(model.parameters()).device

    for i in rng:
        xb = grid[i : i + batch_size].to(device)
        if k == 1:
            s1_batch = top1_sigma_batch(model, xb, layer_spec, cfg)
            lam_batch = torch.log(torch.clamp(s1_batch, min=eps)) / max(int(time_L), 1)
            vals.append(lam_batch.detach().cpu())
        else:
            s1_batch, s2_batch = top2_sigma_batch(model, xb, layer_spec, cfg)
            lam_batch = torch.log(torch.clamp(s2_batch, min=eps)) / max(int(time_L), 1)
            vals.append(lam_batch.detach().cpu())

    return torch.cat(vals, dim=0)

def compute_ftle_grid_both(
    model: torch.nn.Module,
    layer_spec: LayerSpec,
    time_L: int,
    grid: torch.Tensor,
    cfg: SVConfig = SVConfig(exact_if_dim_le=4, jvp_backend="auto"),
    batch_size: int = 1024,
    show_progress: bool = True,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute FTLE for k=1 and k=2 in one pass (slightly more compute, but saves some Python overhead).
    Returns:
      ftle1, ftle2  as [N] tensors on CPU
    """
    model.eval()
    N = grid.size(0)
    vals_1, vals_2 = [], []

    rng = range(0, N, batch_size)
    if show_progress:
        rng = tqdm(rng, total=(N + batch_size - 1) // batch_size, desc="FTLE(k=1,2) grid")

    device = next(model.parameters()).device
    for i in rng:
        xb = grid[i : i + batch_size].to(device)

        s1_batch, s2_batch = top2_sigma_batch(model, xb, layer_spec=layer_spec, cfg=cfg)
        
        lam_1 = torch.log(torch.clamp(s1_batch, min=eps)) / max(int(time_L), 1)
        lam_2 = torch.log(torch.clamp(s2_batch, min=eps)) / max(int(time_L), 1)

        vals_1.append(lam_1.detach().cpu())
        vals_2.append(lam_2.detach().cpu())

    
        # for j in range(xb.size(0)):
        #     x = xb[j]
        #     s1, v1, s2, v2 = top2_sigmas(model, x, layer_spec, cfg)
        #     vals1.append(math.log(max(s1, eps)) / max(int(time_L), 1))
        #     vals2.append(math.log(max(s2, eps)) / max(int(time_L), 1))

    return torch.cat(vals_1, dim=0), torch.cat(vals_2, dim=0)
