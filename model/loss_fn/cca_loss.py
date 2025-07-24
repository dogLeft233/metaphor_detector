import torch

def cca_loss(Hx, Hy, r=1e-3):
    # Hx,Hy: (n, k)
    n, k = Hx.size()
    # 去中心
    Hx = Hx - Hx.mean(dim=0, keepdim=True)
    Hy = Hy - Hy.mean(dim=0, keepdim=True)
    # 协方差
    Cxx = (Hx.T @ Hx) / (n-1) + r * torch.eye(k, device=Hx.device)
    Cyy = (Hy.T @ Hy) / (n-1) + r * torch.eye(k, device=Hx.device)
    Cxy = (Hx.T @ Hy) / (n-1)
    # 白化
    Dx, Vx = torch.linalg.eigh(Cxx)
    Dy, Vy = torch.linalg.eigh(Cyy)
    Cxx_inv_sqrt = Vx @ torch.diag(Dx.clamp(min=1e-12)**-0.5) @ Vx.T
    Cyy_inv_sqrt = Vy @ torch.diag(Dy.clamp(min=1e-12)**-0.5) @ Vy.T
    T = Cxx_inv_sqrt @ Cxy @ Cyy_inv_sqrt
    # SVD
    sv = torch.linalg.svdvals(T)
    # 相关系数之和
    return -sv.sum()