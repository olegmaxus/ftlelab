import torch
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Literal
import numpy as np

SplitDict = Dict[str, Tuple[torch.Tensor, torch.Tensor]]


def split_dataset(
    X, y,
    val_size=0.2,
    val_to_use_as_test=0.4,
    seed=123
) -> SplitDict:

    X_train, X_val_, y_train, y_val_ = train_test_split(X.numpy(), y.numpy(), 
                                                        test_size=val_size,
                                                        random_state=seed,
                                                        stratify=(y.numpy() > 0).astype(int))
    X_val, X_test, y_val, y_test = train_test_split(X_val_, y_val_,
                                                    test_size=val_to_use_as_test,
                                                    random_state=seed,
                                                    stratify=(y_val_ > 0).astype(int))


    return {"train":    (torch.tensor(X_train), torch.tensor(y_train)),
            "val":      (torch.tensor(X_val),   torch.tensor(y_val)),
            "test":     (torch.tensor(X_test),  torch.tensor(y_test))}


def split_features(
    X: torch.Tensor,
    val_size: float = 0.2,
    val_to_use_as_test: float = 0.4,
    seed: int = 123,
    device: str | torch.device | None = None,
) -> Dict[str, torch.Tensor]:
    """
    Train/val/test split for unlabeled feature data (autoencoders).
    """
    X_np = X.detach().cpu().numpy() if isinstance(X, torch.Tensor) else np.asarray(X)

    X_train, X_val_, = train_test_split(
        X_np,
        test_size=val_size,
        random_state=seed,
    )
    X_val, X_test = train_test_split(
        X_val_,
        test_size=val_to_use_as_test,
        random_state=seed,
    )

    splits = {
        "train": torch.as_tensor(X_train, dtype=torch.float32),
        "val": torch.as_tensor(X_val, dtype=torch.float32),
        "test": torch.as_tensor(X_test, dtype=torch.float32),
    }
    if device is not None:
        dev = torch.device(device)
        splits = {k: v.to(dev) for k, v in splits.items()}
    return splits