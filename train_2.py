import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import math
from PIL import Image
import numpy as np
import torch
from torch import nn
from torchvision.datasets import ImageFolder
from torchvision import transforms, models
import torch.nn.functional as F
from torch.autograd import Variable

from efficientnet_pytorch.radam import RAdam
from timm.models.efficientnet import *
from efficientnet_pytorch.utils import (
    round_filters,
    round_repeats,
    drop_connect,
    get_same_padding_conv2d,
    get_model_params,
    efficientnet_params,
    load_pretrained_weights,
    Swish,
    MemoryEfficientSwish,
    calculate_output_image_size
)
from augmentations import *

_IMAGENET_PCA = {
    'eigval': [0.2175, 0.0188, 0.0045],
    'eigvec': [
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ]
}

class EfficientNetRandomCrop:
    def __init__(self, imgsize, min_covered=0.1, aspect_ratio_range=(3./4, 4./3), area_range=(0.08, 1.0), max_attempts=10):
        assert 0.0 < min_covered
        assert 0 < aspect_ratio_range[0] <= aspect_ratio_range[1]
        assert 0 < area_range[0] <= area_range[1]
        assert 1 <= max_attempts

        self.min_covered = min_covered
        self.aspect_ratio_range = aspect_ratio_range
        self.area_range = area_range
        self.max_attempts = max_attempts
        self._fallback = EfficientNetCenterCrop(imgsize)

    def __call__(self, img):
        # https://github.com/tensorflow/tensorflow/blob/9274bcebb31322370139467039034f8ff852b004/tensorflow/core/kernels/sample_distorted_bounding_box_op.cc#L111
        original_width, original_height = img.size
        min_area = self.area_range[0] * (original_width * original_height)
        max_area = self.area_range[1] * (original_width * original_height)

        for _ in range(self.max_attempts):
            aspect_ratio = random.uniform(*self.aspect_ratio_range)
            height = int(round(math.sqrt(min_area / aspect_ratio)))
            max_height = int(round(math.sqrt(max_area / aspect_ratio)))

            if max_height * aspect_ratio > original_width:
                max_height = (original_width + 0.5 - 1e-7) / aspect_ratio
                max_height = int(max_height)
                if max_height * aspect_ratio > original_width:
                    max_height -= 1

            if max_height > original_height:
                max_height = original_height

            if height >= max_height:
                height = max_height

            height = int(round(random.uniform(height, max_height)))
            width = int(round(height * aspect_ratio))
            area = width * height

            if area < min_area or area > max_area:
                continue
            if width > original_width or height > original_height:
                continue
            if area < self.min_covered * (original_width * original_height):
                continue
            if width == original_width and height == original_height:
                return self._fallback(img)      # https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/preprocessing.py#L102

            x = random.randint(0, original_width - width)
            y = random.randint(0, original_height - height)
            return img.crop((x, y, x + width, y + height))

        return self._fallback(img)

class EfficientNetCenterCrop:
    def __init__(self, imgsize):
        self.imgsize = imgsize

    def __call__(self, img):
        """Crop the given PIL Image and resize it to desired size.
        Args:
            img (PIL Image): Image to be cropped. (0,0) denotes the top left corner of the image.
            output_size (sequence or int): (height, width) of the crop box. If int,
                it is used for both directions
        Returns:
            PIL Image: Cropped image.
        """
        image_width, image_height = img.size
        image_short = min(image_width, image_height)

        crop_size = float(self.imgsize) / (self.imgsize + 32) * image_short

        crop_height, crop_width = crop_size, crop_size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))
        return img.crop((crop_left, crop_top, crop_left + crop_width, crop_top + crop_height))

def train(arch, model, dataloaders, dataset_size, criterion, optimizer, num_epochs, valid_loss_min = np.Inf):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(device)
    model = model.to(device)
    criterion = criterion.to(device)
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs), '\n-------------------------------')
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0
            accuracy = 0
            for features, targets in dataloaders[phase]:
                features = features.to(device)
                targets = targets.to(device)
                
                optimizer.zero_grad()
                outputs = model(features)
                _, preds = torch.max(outputs, 1)
                accuracy += torch.sum(preds == targets.data)
                # print(targets.size())
                # print(outputs.size())
                loss = criterion(outputs, targets)
                # print('----Loss', loss)
                # print('----Features', features.size())
                running_loss += loss.item() * features.size(0)
                # print('----Running loss', running_loss)
                
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, running_loss / dataset_size[phase], accuracy.double() / dataset_size[phase]))
            if phase == 'val' and running_loss <= valid_loss_min:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                    valid_loss_min,
                    running_loss))
                torch.save(model.state_dict(), 'model_' + arch + '.pt')
                valid_loss_min = running_loss
    return model, valid_loss_min

if __name__ == "__main__":
    model = efficientnet_b0(True)
    print(model)
    model.classifier = nn.Linear(1280, 42)
    print(model)

    data_dir = './dataset/train'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/val'

    input_size = 224
    sized_size = 256

    data_transforms = {
        "train": transforms.Compose([
            EfficientNetRandomCrop(sized_size),
            transforms.Resize((input_size, input_size), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
            ),
            transforms.ToTensor(),
            Lighting(0.1, _IMAGENET_PCA['eigval'], _IMAGENET_PCA['eigvec']),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            EfficientNetCenterCrop(sized_size),
            transforms.Resize((input_size, input_size), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }

    image_datasets = {
        "train": ImageFolder(train_dir, transform=data_transforms['train']),
        "val": ImageFolder(valid_dir, transform=data_transforms['val'])
    }

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {
        "train": torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True, pin_memory=True, num_workers=8),
        "val": torch.utils.data.DataLoader(image_datasets['val'], batch_size=8, shuffle=True, pin_memory=True, num_workers=8)
    }

    dataset_size = {
        'train': len(image_datasets['train']),
        'val': len(image_datasets['val']),
    }


    criterion = nn.CrossEntropyLoss()
    optimizer = RAdam(model.parameters())
    train('efficient-2-b0', model, dataloaders, dataset_size, criterion, optimizer, 100)