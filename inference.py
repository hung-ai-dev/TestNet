import os
import os.path as osp
import glob
from PIL import Image

import torch
from torchvision import transforms, models
from torch import nn
import pandas as pd

from timm.models.efficientnet import *

def convert(img):
    image_size = 224
    tfms = transforms.Compose([
            transforms.Resize(image_size), 
            # transforms.RandomRotation(10),
            # transforms.Crop(image_size), 
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
    img = tfms(img).unsqueeze(0)
    return img

if __name__ == "__main__":
    path = './dataset/test/test'
    model = tf_efficientnet_b1_ns(True)
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
        # vote = [0 for i in range(42)]
        # for i in range(10):
        img = Image.open(img_path)
        img = convert(img).cuda()  
        with torch.no_grad():
            logits = model(img)
        pred = torch.topk(logits, k=1).indices.squeeze(0).tolist()[0]
            # vote[pred] += 1

        res['filename'].append(osp.basename(img_path))
        res['category'].append("{:02d}".format(pred))
        print(osp.basename(img_path), '---', pred)
    print(res)
    df = pd.DataFrame(res)
    df.to_csv("res_B0NS.csv", index=False)