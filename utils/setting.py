import torch
import random
import numpy as np

def set_random_seed(random_seed: int):
    """
    设置随机种子以确保实验的可重复性
    
    该函数会设置PyTorch、Python random、NumPy和CUDA的随机种子，
    并启用CUDA的确定性模式，确保每次运行得到相同的结果。
    
    Args:
        random_seed (int): 随机种子值，建议使用固定值如42
        
    Note:
        - 设置CUDA确定性模式可能会影响性能，但能保证结果一致性
        - 在生产环境中，如果不需要完全确定性，可以注释掉CUDA相关设置
    """
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False