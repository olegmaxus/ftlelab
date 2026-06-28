import torch


def device_string() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_device(device: str | torch.device | None = None) -> torch.device:
    if device is None:
        return torch.device(device_string())
    return torch.device(device)


def to_device(x, device: str | torch.device | None = None):
    """
    Move tensor(s) to the requested device. Accepts a tensor, list/tuple, or dict.
    """
    dev = get_device(device)

    if isinstance(x, torch.Tensor):
        return x.to(dev)
    if isinstance(x, (tuple, list)):
        return type(x)(to_device(v, dev) for v in x)
    if isinstance(x, dict):
        return {k: to_device(v, dev) for k, v in x.items()}
    return x
