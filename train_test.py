import glob
import os
import os.path as osp
import shutil

train_path = 'shopee-product-detection-dataset-002/train/train'
val_path = 'shopee-product-detection-dataset-002/train/val'

sub_fol = os.listdir(train_path)
print(sub_fol)
for sub in sub_fol:
    val_fol = osp.join(val_path, sub)
    if not osp.exists(val_fol):
        os.makedirs(osp.join(val_path, sub))
    train_fol = osp.join(train_path, sub)
    all_images = glob.glob(train_fol + '/*.jpg')

    for idx, img_path in enumerate(all_images):
        if idx > len(all_images) * 0.2:
            break
        shutil.move(img_path, val_fol)