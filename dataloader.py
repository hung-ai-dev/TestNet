import os
import os.path as osp
import collections

from PIL import Image
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils import data
import torchvision
from torchvision import transforms, models

from efficientnet_pytorch.utils import *

class ProductImageLoader(data.Dataset):
    def __init__(self, root, data_path, split, is_transform=False, n_classes=42):
        self.is_transform = is_transform
        self.n_classes = n_classes
        self.data_path = data_path
        self.split = split
        df = pd.read_csv(data_path, header=None)
        all_img_name = df[0].to_list()
        all_category = df[1].to_list()
        self.files = all_img_name #[osp.join(root, img_name) for img_name in all_img_name]
        self.classes = all_category

    def __len__(self):
        return len(self.files)
    
    def __transform(self, img):
        sized_size = 256
        image_size = 224

        if self.split == 'val':
            tfms = transforms.Compose([
                            transforms.Resize(image_size), transforms.CenterCrop(image_size), 
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        else:
            tfms = transforms.Compose([
                            EfficientNetRandomCrop(sized_size),
                            transforms.Resize((image_size, image_size), interpolation=Image.BICUBIC),
                            transforms.RandomHorizontalFlip(),
                            transforms.ColorJitter(
                                brightness=0.4,
                                contrast=0.4,
                                saturation=0.4,
                            ),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        img = tfms(img)
        return img

    def __getitem__(self, index):
        img = Image.open(self.files[index])
        lbl = self.classes[index]

        img = self.__transform(img)

        return img, lbl