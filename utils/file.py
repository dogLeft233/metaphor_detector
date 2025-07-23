from PIL import Image
import numpy as np
import torch
from typing import Tuple

def read_image(image_path, shape:Tuple[int, int, int] = (1, 1, 3))->torch.Tensor:
    
    try:
        img = Image.open(image_path)
        if img.mode == "L":
            img = img.convert("RGB")
        img = np.array(img)
    except:
        img = np.zeros(shape)
        
    return torch.tensor(img).permute(2, 0, 1)