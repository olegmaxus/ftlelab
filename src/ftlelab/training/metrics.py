import torch

__all__ = [
    "binary_accuracy",
    "multiclass_accuracy",
    "mse_metric",
    "reconstruction_error",
    "vae_kl_divergence",
]


def binary_accuracy(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    mode: str = "mse"
) -> float:
    
    with torch.no_grad():
        if mode == "mse":
            pred = torch.sign(outputs)
            targ = torch.sign(targets)
            return (pred == targ).float().mean().item()
        
        if mode == "bce_logits":
            pred = (outputs >= 0.0).float()
            return (pred == targets.float()).float().mean().item()

        if mode == "bce":
            pred = (outputs >= 0.5).float()
            return (pred == targets.float()).float().mean().item()
        
        raise ValueError(f"Unknown binary metric mode: {mode}")

def multiclass_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    with torch.no_grad():
        pred = logits.argmax(dim=-1)
        targ = targets.long().view(-1)
        return (pred == targ).float().mean().item()

def mse_metric(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    with torch.no_grad():
        return ((outputs - targets) ** 2).mean().item()

def reconstruction_error(recon: torch.Tensor, target: torch.Tensor) -> float:
    with torch.no_grad():
        return ((recon - target) ** 2).mean().item()

def vae_kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> float:
    """
    Mean KL divergence term for a VAE.
    """
    with torch.no_grad():
        kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        return kl.item()