from scipy.special import gamma

def bandwidth(d):
    gz = 2 * gamma(0.5 * (d+1)) / gamma(0.5 * d)
    return 1. / (2. * gz**2)

def knl(x, y, gam=1.):
    dist_table = (x.unsqueeze(0) - y.unsqueeze(1)).pow(2).sum(dim = 2)
    return (-gam * dist_table).exp().transpose(0,1)

def hsic(x, y):
    dx = x.shape[1]
    dy = y.shape[1]

    xx = knl(x, x, gam=bandwidth(dx))
    yy = knl(y, y, gam=bandwidth(dy))

    res = ((xx*yy).mean()) + (xx.mean()) * (yy.mean())
    res -= 2*((xx.mean(dim=1))*(yy.mean(dim=1))).mean()
    return res.clamp(min = 1e-16).sqrt()
