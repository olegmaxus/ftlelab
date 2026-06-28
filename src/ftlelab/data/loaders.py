from dataclasses import dataclass
from typing import Dict, Tuple, Literal

import torch
from torch.utils.data import DataLoader, TensorDataset

SplitDict = Dict[str, Tuple[torch.Tensor, torch.Tensor]]


def make_dataloaders(
    splits: SplitDict,
    batch_size: int = 256,
    shuffle_train: bool = True,
    drop_last: bool = False,
    num_workers: int = 0,
) -> Dict[str, DataLoader]:
    """
    Turn a {train,val,test} split dict into PyTorch DataLoaders.

    Each split entry is (X, y) as torch tensors.
    """
    loaders: Dict[str, DataLoader] = {}

    for split_name, (X, y) in splits.items():
        ds = TensorDataset(X, y)
        shuffle = shuffle_train if split_name == "train" else False
        loaders[split_name] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last if split_name == "train" else False,
            num_workers=num_workers,
        )

    return loaders


def make_feature_dataloaders(
    splits: Dict[str, torch.Tensor],
    batch_size: int = 256,
    shuffle_train: bool = True,
    drop_last: bool = False,
    num_workers: int = 0,
) -> Dict[str, DataLoader]:
    """
    Turn a {train,val,test} split dict of feature tensors into DataLoaders.
    Each batch is a single tensor (x,) suitable for autoencoder training.
    """
    loaders: Dict[str, DataLoader] = {}

    for split_name, X in splits.items():
        ds = TensorDataset(X)
        shuffle = shuffle_train if split_name == "train" else False
        loaders[split_name] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last if split_name == "train" else False,
            num_workers=num_workers,
        )

    return loaders


@dataclass
class MNISTWrapper:
    """
    Placeholder for a future MNIST/image dataset integration.

    For now, this could later:
    - download/load MNIST,
    - apply transforms,
    - expose train/val/test splits and loaders.
    """
    root: str = "./data"
    download: bool = True

    def prepare(self):
        raise NotImplementedError(
            "MNISTWrapper is a placeholder; integrate torchvision or "
            "your own loader here when you need MNIST/image experiments."
        )