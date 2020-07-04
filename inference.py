import os
import os.path as osp
import glob
from PIL import Image

import torch
from torchvision import transforms, models
from torch import nn
import pandas as pd

from timm.models.efficientnet import *
from efficientnet_pytorch.utils import *

def convert(img):
    sized_size = 256
    image_size = 224
    tfms = transforms.Compose([
        EfficientNetRandomCrop(sized_size),
        transforms.Resize((image_size, image_size), interpolation=Image.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
        ),
        transforms.RandomRotation(10, resample=Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    img = tfms(img).unsqueeze(0)
    return img

if __name__ == "__main__":
    path = './dataset/test/test'
    model = tf_efficientnet_b1_ns(False)
    model.classifier = nn.Linear(1280, 42)
    model.load_state_dict(torch.load('./model_B0NS.pt'))
    model.cuda().eval()

    test_df = pd.read_csv('./dataset/test.csv')
    all_img_name = test_df['filename'].to_list()
    # print(image_name)

    res = {'filename': [],
            'category': []}

    # all_images = glob.glob(osp.join(path, '*.jpg'))
    for img_name in all_img_name:
        img_path = osp.join('dataset/test/test', img_name)
        vote = [0 for i in range(42)]
        for i in range(15):
            img = Image.open(img_path)
            img = convert(img).cuda()  
            with torch.no_grad():
                logits = model(img)
            pred = torch.topk(logits, k=1).indices.squeeze(0).tolist()[0]
            vote[pred] += 1

        res['filename'].append(osp.basename(img_path))
        res['category'].append("{:02d}".format(vote.index(max(vote))))
        print(osp.basename(img_path), '---', vote.index(max(vote)))
    print(res)
    df = pd.DataFrame(res)
    df.to_csv("res_B0NS.csv", index=False)