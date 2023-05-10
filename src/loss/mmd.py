import torch
import numpy as np

def fast_mmd(x, y, D: int = 500):
    gamma = 1
    W = torch.randn(x.shape[1], D, device=x.device)
    b = 2*np.pi*torch.rand(D, device=x.device)

    psi_x = np.sqrt(2./D)*torch.cos(np.sqrt(2./gamma)*torch.matmul(x, W) + b)
    psi_y = np.sqrt(2./D)*torch.cos(np.sqrt(2./gamma)*torch.matmul(y, W) + b)

    diff = torch.mean(psi_x, dim=0) - torch.mean(psi_y, dim=0)
    return torch.sum(diff**2)
