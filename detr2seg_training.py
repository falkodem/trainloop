import os
import argparse
import yaml

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from transformers import DetrConfig, DetrForSegmentation
import albumentations as A

from TrainLoop.dataset.instance_seg import InstanceSegDETR, DetrPreprocessor

optimizer_mapping = {'AdamW': torch.optim.AdamW}
scheduler_mapping = {'ExponentialLR': torch.optim.lr_scheduler.ExponentialLR,
                     'CosineAnnealing': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts}

def save_random_train_imgs(ds, save_path):
    os.makedirs(save_path, exist_ok=True)
    f, ax = plt.subplots(4,4, figsize=(12,7))
    np.random.seed(3)
    random_idxs = np.random.choice(np.arange(len(ds)), 16, replace=False)
    for i, idx in enumerate(random_idxs):
        data = ds[idx]
        img = np.array(data['pixel_values']).transpose(1,2,0)
        img = img + np.abs(img.min(axis=(0,1)))
        img = img * (255 / img.max(axis=(0,1)))
        ann = np.max(np.array(data['labels']['masks']), axis=0)
        ann = ann * (255 / ann.max())
        res = (img*0.5).astype(np.uint8) + (np.repeat(np.expand_dims(ann, axis=2), 3, axis=2)*0.5).astype(np.uint8)
        bboxes = np.array(data['labels']['boxes'])
        for box in bboxes:
            x, y, w, h = box.astype(int)
            rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='g', facecolor='none')
            ax[i//4, i%4].add_patch(rect)
        ax[i//4, i%4].imshow(res)
        ax[i//4, i%4].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'train_img_examples.jpg'), dpi=300)
    plt.clf()
    plt.close()


def main(args):
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    

    # ---------- Data settings ----------
    size = cfg['preprocessor']['size']

    augmentations = A.Compose([
        # A.CropNonEmptyMaskIfExists(width=cfg['aug']['crop_size'][0], height=cfg['aug']['crop_size'][1]),
        A.Perspective(scale=[*cfg['aug']['perspective']['scale']], keep_size=True),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=cfg['aug']['brightness_contrast']['p']),
        A.Rotate(limit=[*cfg['aug']['rotate']['angles']]),
        A.Resize(height=size[0], width=size[1])
        ],
    bbox_params=A.BboxParams(format='coco'))

    preprocessor_train = DetrPreprocessor(cfg['preprocessor']['size'], augmentator=augmentations)
    preprocessor_val = DetrPreprocessor(cfg['preprocessor']['size'])

    subset = 'train'
    ds_train = InstanceSegDETR(f'data/car_damage_small/{subset}',
                                               f'data/car_damage_small/{subset}/_annotations.coco.json',
                                               preprocessor_train)
    subset = 'valid'
    ds_val = InstanceSegDETR(f'data/car_damage_small/{subset}',
                                             f'data/car_damage_small/{subset}/_annotations.coco.json',
                                             preprocessor_val)
    subset = 'test'
    ds_test = InstanceSegDETR(f'data/car_damage_small/{subset}',
                                              f'data/car_damage_small/{subset}/_annotations.coco.json',
                                              preprocessor_val)

    def collate_fn(batch):
        processed_data = {}
        processed_data['pixel_values'] = [b['pixel_values'] for b in batch]
        processed_data['pixel_masks'] = [b['pixel_masks'] for b in batch]
        processed_data['labels'] = [b['labels'] for b in batch]
        return processed_data

    dl_train = DataLoader(ds_train, batch_size=cfg['dataset']['batch_size'], shuffle=True, num_workers=cfg['dataset']['workers'], collate_fn=collate_fn)
    dl_val = DataLoader(ds_val, batch_size=cfg['dataset']['batch_size'], shuffle=False, num_workers=cfg['dataset']['workers'], collate_fn=collate_fn)


    # print(next(iter(dl_train)))
    # a = ds_train[1]
    # ---------- Create Model ----------
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device:', device)
    save_random_train_imgs(ds_train, os.path.join('runs', cfg['run_name']))
    # model = DetrForSegmentation.from_pretrained("facebook/detr-resnet-50-panoptic")
    # optimizer = optimizer_mapping[cfg['train']['optimizer']['name']](model.parameters(), **cfg['train']['optimizer']['params'])
    # scheduler = scheduler_mapping[cfg['train']['scheduler']['name']](optimizer, **cfg['train']['scheduler']['params'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    
    args = parser.parse_args()
    
    main(args)