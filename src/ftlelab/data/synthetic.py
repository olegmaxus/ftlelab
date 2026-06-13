import torch
from typing import Tuple

from ..utils import set_seed
import math

TensorPair = Tuple[torch.Tensor, torch.Tensor]

def make_moons_dataset(
    num_samples: int = 10000,
    noise_std: float = 0.05,
    seed: int | None = 123,
    shuffle: bool = True,
    radius: float = 1.0,
    offset: tuple[float, float] = (1.0, 0.5)
) -> TensorPair:
    if seed is not None:
        set_seed(seed)

    n_plus = num_samples // 2
    n_minus = num_samples - n_plus

    t_plus = torch.rand(n_plus) * math.pi
    t_minus = torch.rand(n_minus) * math.pi

    r = torch.tensor(radius)
    ox = torch.tensor(offset[0])
    oy = torch.tensor(offset[1])

    x_plus = torch.stack([r * torch.cos(t_plus),
                          r * torch.sin(t_plus)], dim=1)
    x_minus = torch.stack([ox - r * torch.cos(t_minus),
                           oy - r * torch.sin(t_minus)], dim=1)

    X = torch.cat([x_plus, x_minus], dim=0)
    y = torch.cat([torch.ones(n_plus, 1),
                   -torch.ones(n_minus, 1)], dim=0)

    # Add isotropic Gaussian noise
    if noise_std and noise_std > 0:
        X = X + torch.randn(X.shape) * noise_std

    if shuffle:
        idx = torch.randperm(X.size(0))
        X, y = X[idx], y[idx]

    return X, y


def make_circle_dataset(
    num_samples: int = 10000,
    radius: float = 0.5,
    noise_std: float = 0.01, 
    seed: int = 123,
    margin: float = 0.0,
) -> TensorPair:
    set_seed(seed)

    half_margin = margin / 2.0
    extent = 2.0 * radius
    X_list = []
    y_list = []
    collected = 0

    while collected < num_samples:
        batch_size = num_samples - collected + int(0.2 * num_samples)
        X_batch = (torch.rand(batch_size, 2) * 2 - 1) * extent

        distance = torch.linalg.norm(X_batch, dim=1)

        inside_mask = distance < (radius - half_margin)
        outside_mask = distance > (radius + half_margin)
        valid_mask = inside_mask | outside_mask

        X_valid = X_batch[valid_mask]
        y_valid = torch.where(inside_mask[valid_mask],  1., -1.).reshape(-1, 1)
        
        X_list.append(X_valid)
        y_list.append(y_valid)
        collected += X_valid.size(0)

    X = torch.cat(X_list, dim=0)[:num_samples]
    y = torch.cat(y_list, dim=0)[:num_samples]

    if noise_std and noise_std > 0.0:
        X = X + torch.normal(0, noise_std, size=X.shape)

    return X, y


def make_spiral_dataset(
    num_samples: int = 10000,
    noise_std: float = 0.01,
    seed: int = 123,
) -> TensorPair:
    set_seed(seed)
    def spiral(n, delta):
        t = torch.rand(n,) * 4 * torch.pi
        x = t * torch.cos(t + delta) + torch.normal(0, noise_std, size=(n,))
        y = t * torch.sin(t + delta) + torch.normal(0, noise_std, size=(n,))
        return torch.stack([x, y], dim=1)
    
    Xp = spiral(num_samples // 2, 0.0)
    Xn = spiral(num_samples - num_samples // 2, torch.pi)
    
    X = torch.vstack([Xp, Xn])
    y = torch.hstack([torch.ones(num_samples // 2), -torch.ones(num_samples - num_samples // 2)])
    
    X = (X - X.mean(0)) / X.std(0)
    
    return X, y.reshape(-1, 1)


def make_xor_dataset(
    num_samples: int = 10000,
    noise_std: float = 0.01,
    seed: int = 123,
    margin: float = 0.0, 
) -> TensorPair:
    set_seed(seed)

    half_margin = margin / 2.0
    X_list = []
    y_list = []
    collected = 0

    while collected < num_samples:
        batch_size = num_samples - collected + int(0.2 * num_samples)
        X_batch = torch.rand(batch_size, 2) * 2 - 1

        valid_mask = (X_batch[:, 0].abs() > half_margin) & (X_batch[:, 1].abs() > half_margin)

        X_valid = X_batch[valid_mask]
        y_valid = torch.where((X_valid[:, 0] * X_valid[:, 1]) > 0, 1., -1.).reshape(-1, 1)
        
        X_list.append(X_valid)
        y_list.append(y_valid)
        collected += X_valid.size(0)

    X = torch.cat(X_list, dim=0)[:num_samples]
    y = torch.cat(y_list, dim=0)[:num_samples]

    if noise_std and noise_std > 0.0:
        X = X + torch.normal(0, noise_std, size=X.shape)
        
    return X, y

