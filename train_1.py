import argparse
import os
import sys

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import math
from PIL import Image
import numpy as np
import torch
from torch import nn
from torchvision.datasets import ImageFolder
from torchvision import transforms, models
import torch.nn.functional as F
from torch.autograd import Variable

from tqdm import tqdm

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
    calculate_output_image_size,
)
from augmentations import *
from cyclic_lr import CyclicLR
from dataloader import ProductImageLoader

_IMAGENET_PCA = {
    "eigval": [0.2175, 0.0188, 0.0045],
    "eigvec": [
        [-0.5675, 0.7192, 0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948, 0.4203],
    ],
}

class SCELoss(torch.nn.Module):
    def __init__(self, alpha, beta, num_classes=10):
        super(SCELoss, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, pred, labels):
        # CCE
        ce = self.cross_entropy(pred, labels)

        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))

        # Loss
        loss = self.alpha * ce + self.beta * rce.mean()
        return loss

def train(
    arch,
    model,
    dataloaders,
    dataset_size,
    criterion,
    optimizer,
    scheduler,
    num_epochs,
    valid_loss_min=np.Inf,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    model = model.to(device)
    # criterion = criterion.to(device)
    iters = len(dataloaders['train'])

    for epoch in range(num_epochs):
        print(
            "Epoch {}/{}".format(epoch, num_epochs), "\n-------------------------------"
        )

        for phase in ["val", "train"]:
            if phase == "train":
                model.train()
            else:
                model.eval()
            running_loss = 0
            accuracy = 0
            idx = 0
            for features, targets in tqdm(dataloaders[phase]):
                features = features.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()
                outputs = model(features)
                _, preds = torch.max(outputs, 1)
                accuracy += torch.sum(preds == targets.data)
                loss = criterion(outputs, targets)
                # print(loss)
                running_loss += loss.item() * features.size(0)

                if phase == "train":
                    loss.backward()
                    optimizer.step()
                    idx += 1
                    scheduler.batch_step()
            print(
                "{} Loss: {:.4f} Acc: {:.4f}".format(
                    phase,
                    running_loss / dataset_size[phase],
                    accuracy.double() / dataset_size[phase],
                )
            )
            torch.save(model.state_dict(), "lastest_" + arch + '.pt')
            if phase == "val" and running_loss <= valid_loss_min:
                print(
                    "Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...".format(
                        valid_loss_min, running_loss
                    )
                )
                torch.save(model.state_dict(), "model_" + arch + ".pt")
                valid_loss_min = running_loss
    return model, valid_loss_min


if __name__ == "__main__":
    arch = sys.argv[1]
    batch_size = int(sys.argv[2])
    print(arch)
    if arch == "B6NS":
        model = tf_efficientnet_b6_ns(True)
        model.classifier = nn.Linear(2304, 42)
    elif arch == "B7NS":
        model = tf_efficientnet_b7_ns(True)
        model.classifier = nn.Linear(2560, 42)
    elif arch == "B1NS":
        model = tf_efficientnet_b1_ns(True)
        print(model)
        model.classifier = nn.Linear(1280, 42)
    elif arch == "B5NS":
        model = tf_efficientnet_b5_ns(True)
        model.classifier = nn.Linear(2048, 42)
    elif arch == "B0NS":
        model = tf_efficientnet_b1_ns(True)
        print(model)
        model.classifier = nn.Linear(1280, 42)

    # print(model)
    # model.load_state_dict(torch.load('./model_B1NS.pt'))
    # print(model)
    for param in model.parameters():
        param.requires_grad = True

    train_set = ProductImageLoader(None, "./dataset/train/fold0_train.csv", "train")
    val_set = ProductImageLoader(None, "./dataset/train/fold0_test.csv", "val")

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {
        "train": torch.utils.data.DataLoader(
            train_set,
            batch_size=int(batch_size * 1.5),
            shuffle=True,
            pin_memory=False,
            num_workers=8,
        ),
        "val": torch.utils.data.DataLoader(
            val_set,
            batch_size=batch_size // 2,
            shuffle=False,
            pin_memory=False,
            num_workers=8,
        ),
    }

    dataset_size = {"train": len(train_set), "val": len(val_set)}

    criterion = SCELoss(1.0, 1.0, 42) #nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)
    scheduler = CyclicLR(optimizer, base_lr=0.00001, max_lr=0.01, step_size=2000, last_batch_iteration=1)
    train(arch, model, dataloaders, dataset_size, criterion, optimizer, scheduler, 100)