import torch

def kl_gaussian(logvar_a, mu_a, logvar_b, mu_b): #(B, D) * 4
    per_kl = logvar_b - logvar_a - 1 + (logvar_a.exp() + (mu_a - mu_b).square()) / logvar_b.exp()
    kl = 0.5 * torch.sum(per_kl, dim=1)
    return kl

def kl_bernoulli(p1, p2): #(B) * 2
    kl = p1 * (torch.log(p1) - torch.log(p2)) + (1-p1) * (torch.log(1-p1) - torch.log(1-p2))
    return kl
