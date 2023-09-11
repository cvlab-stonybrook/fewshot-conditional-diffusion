import glob
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset
import torchvision.transforms.functional as tF

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Colormap
n_classes = 19
tab20_colors = plt.cm.tab20(np.arange(n_classes))

# It seems like lower-upper lip is not labeled
mask_cmap_colors = np.array([
    tab20_colors[0], # 0    
    tab20_colors[16], # 1
    tab20_colors[9], # 2
    tab20_colors[6], # 3
    tab20_colors[17], # 4
    tab20_colors[18], # 5
    tab20_colors[2], # 6
    tab20_colors[7], # 7
    tab20_colors[4], # 8
    tab20_colors[13], # 9
    tab20_colors[11], # 10
    tab20_colors[14], # 11
    [0, 0, 0, 1.], # 12
    tab20_colors[10], # 13
    tab20_colors[3], # 14
    tab20_colors[8], # 15
    tab20_colors[5], # 16
    tab20_colors[1], # 17
    [0, 0.0001, 0, 1.], # 18
    [1, 0.000001, 0, 1.], # 19
])
mask_cmap = ListedColormap(mask_cmap_colors)
assert len(np.unique(mask_cmap_colors, axis=1)) == len(mask_cmap_colors)

class FewShotDataset(Dataset):
    def __init__(self, path, p_flip=0.5, p_brightness=0.2, augment=True, resize=None):
        self.path = path
        self.augment = augment
        self.p_flip = p_flip
        self.p_brightness = p_brightness
        self.resize = resize
        
        self.img_paths = glob.glob(path + '*.jpg')
        self.img_type = '.jpg'
        if len(self.img_paths) == 0:
            self.img_paths = glob.glob(path + '*.png')
            self.img_type = '.png'
        self.img_paths.sort()
        self.n_images = len(self.img_paths)
        print(f"Loaded {self.n_images} images from '{self.path}'")
        
    def __len__(self):
        return self.n_images
    
    def __getitem__(self, idx):
        img = torchvision.io.read_image(self.img_paths[idx]) / 255.
        mask = np.load(self.img_paths[idx].replace(self.img_type, '.npy'), allow_pickle=True)
        mask = torch.from_numpy(mask)
            
        # Augmentations 
        if self.augment:
            if np.random.rand() < self.p_flip:
                img = tF.hflip(img)
                mask = tF.hflip(mask)
                
            if np.random.rand() < self.p_brightness:
                img = tF.adjust_brightness(img, 0.5 + 1*np.random.rand())

        if self.resize is not None:
            img = tF.resize(img, self.resize)
            mask_h, mask_w = mask.shape[-2:]
            mask = F.interpolate(mask.view(1,1,mask_h,mask_w), self.resize, mode='nearest').view(self.resize)
        
        return img, mask

