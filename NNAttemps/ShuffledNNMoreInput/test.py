import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import json
import numpy as np
import OpenEXR
import Imath
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

# Function to read EXR file
def read_exr(file_path):
    exr_file = OpenEXR.InputFile(file_path)
    dw = exr_file.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    
    # Read the RGB channels
    rgb = [np.frombuffer(exr_file.channel(c, FLOAT), dtype=np.float32) for c in "RGB"]
    rgb = [np.reshape(c, (size[1], size[0])) for c in rgb]  # Note the order of size
    
    # Stack the channels to form an (H, W, 3) array
    stacked_rgb = np.stack(rgb, axis=-1)
    return size, stacked_rgb

color_exr_path = "C:/Files/CGProject/NNLightProbes/dumped_data/temptemp/processed_real/val/batch_4/color.exr"
nums = 4104
np.set_printoptions(threshold=np.inf, linewidth=np.inf)
_, color_image = read_exr(color_exr_path)
flat_colors = color_image.reshape(-1, 3)
print(flat_colors[:100])