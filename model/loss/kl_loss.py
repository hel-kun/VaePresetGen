import torch
import torch.nn as nn


class KLDivergenceLoss(nn.Module):
    """
    KL(q||p) for diagonal Gaussians: q ~ N(mu_q, var_q), p ~ N(mu_p, var_p)
    """

    def __init__(self):
        super().__init__()

    def forward(self, mu_q: torch.Tensor, logvar_q: torch.Tensor, mu_p: torch.Tensor, logvar_p: torch.Tensor):
        var_q = torch.exp(logvar_q)
        var_p = torch.exp(logvar_p)
        kl = 0.5 * (
            torch.log(var_p) - torch.log(var_q) + (var_q + (mu_q - mu_p) ** 2) / var_p - 1
        )
        return kl.mean()

