import torch
import numpy as np
from sklearn.model_selection import train_test_split
from .utils import set_seed
import math

import math
import torch

def make_moons_dataset(
    n_samples: int = 10000,
    noise_std: float = 0.05,
    seed: int | None = 123,
    shuffle: bool = True,
    radius: float = 1.0,
    offset: tuple[float, float] = (1.0, 0.5)
):
    if seed is not None:
        set_seed(seed)

    n_plus = n_samples // 2
    n_minus = n_samples - n_plus

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
    num_samples=10000, 
    radius=0.5, 
    noise_std=0.01, 
    seed=123
):
    set_seed(seed)

    X = torch.rand(num_samples, 2) * 8 * radius - 4 * radius 
    y = (torch.where(torch.linalg.norm(X, axis=1) < math.sqrt(2 * radius), 1., -1.)).reshape(-1, 1)

    if noise_std and noise_std > 0.0:
        X = X + torch.normal(0, noise_std, size=X.shape)

    return X, y

def make_spiral_dataset(num_samples=10000, noise_std=0.01, seed=123):
    set_seed(seed)
    def spiral(n, delta):
        t = torch.rand(n,) * 4 * torch.pi
        x = t * torch.cos(t + delta) + torch.normal(0, noise_std, size=(n,))
        y = t * torch.sin(t + delta) + torch.normal(0, noise_std, size=(n,))
        return torch.stack([x, y], axis=1)
    
    Xp = spiral(num_samples // 2, 0.0)
    Xn = spiral(num_samples - num_samples // 2, torch.pi)
    
    X = torch.vstack([Xp, Xn])
    y = torch.hstack([torch.ones(num_samples // 2), -torch.ones(num_samples - num_samples // 2)])
    
    X = (X - X.mean(0)) / X.std(0)
    
    return X, y

def split_dataset(X, y, val_size=0.2, val_to_use_as_test=0.4, seed=123):

    X_train, X_val_, y_train, y_val_ = train_test_split(X.numpy(), y.numpy(), 
                                                        test_size=val_size,
                                                        random_state=seed,
                                                        stratify=(y.numpy() > 0).astype(int))
    X_val, X_test, y_val, y_test = train_test_split(X_val_, y_val_,
                                                    test_size=val_to_use_as_test,
                                                    random_state=seed,
                                                    stratify=(y_val_ > 0).astype(int))


    return {"train": (torch.tensor(X_train), torch.tensor(y_train)),
            "val": (torch.tensor(X_val), torch.tensor(y_val)),
            "test": (torch.tensor(X_test), torch.tensor(y_test))}

class MNISTWrapper:
    # Placeholder for later extension if needed
    pass
