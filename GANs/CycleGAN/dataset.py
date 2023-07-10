"""
数据收集和准备：收集用于两个域之间转换的图像数据集，并将其分为两个域的子集。
确保每个域的图像数量相近，以便训练过程更加稳定。
同时，进行必要的数据预处理，如调整大小、裁剪或标准化。

"""

from PIL import Image
import os
import numpy as np
from torch.utils.data import Dataset 

class HorseZerbraDataset(Dataset):
    """
    There are two different datasets(horse and zebra)
    horse_dir: the path of horse images
    zebra_dir: the path of zebra images
    """
    def __init__(self, horse_dir,zebra_dir, transform=None):
        self.horse_dir = horse_dir
        self.zebra_dir = zebra_dir
        self.transform = transform

        self.horse_images = os.listdir(horse_dir)
        self.zebra_images = os.listdir(zebra_dir)
        self.length_dataset = max(len(self.horse_images),len(self.zebra_images))
        self.zebra_len = len(self.zebra_images)
        self.horse_len = len(self.horse_images)
        

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        zebra_img = self.zebra_images[index % self.zebra_len]
        horse_img = self.horse_images[index % self.horse_len]

        zebra_path = os.path.join(self.zebra_dir,zebra_img)
        horse_path = os.path.join(self.horse_dir,horse_img)
        
        zebra_img = np.array(Image.open(zebra_path).convert("RGB"))
        horse_img = np.array(Image.open(horse_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=zebra_img,image0=horse_img)
            zebra_img = augmentations["image"]
            horse_img = augmentations["image0"]

        return horse_img,zebra_img

