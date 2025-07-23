import torch
import random
import numpy as np

def set_random_seed(random_seed:int):
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
    
    # 设置 CUDA 的确定性模式
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False