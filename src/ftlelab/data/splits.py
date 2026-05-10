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