# src/ftlelab/data/viz.py

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import torch


def plot_2d_dataset(
    X: torch.Tensor,
    y: torch.Tensor,
    title: str = "2D dataset",
    colors: tuple[str, str] = ("#1d3578", "#8f7e33"),
):
    """
    Scatter plot for 2D binary datasets with labels in {-1, +1}.

    X: (N, 2)
    y: (N,) or (N, 1) with values -1 / +1.
    """
    X = X.detach().cpu()
    y = y.detach().cpu().view(-1)

    cmap = ListedColormap(list(colors))

    plt.figure(figsize=(5, 5), dpi=250)
    sc = plt.scatter(
        X[:, 0],
        X[:, 1],
        c=y,
        cmap=cmap,
        s=10,
        alpha=0.8,
        edgecolors="none",
    )
    plt.gca().set_aspect("equal", "box")
    cbar = plt.colorbar(sc, ticks=[-1, 1], fraction=0.046, pad=0.04)
    cbar.set_label("Classes")
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.title(title)
    plt.tight_layout()
    plt.show()

    
def plot_ftle_field():
    pass

def plot_sigma_field():
    pass

def plot_training_curves():
    pass

def plot_ridge_overlay():
    pass

