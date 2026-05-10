import torch

def device_string():
    return "cuda" if torch.cuda.is_available() else "cpu"