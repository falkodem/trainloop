import os
import json
import argparse
import yaml
import joblib

import pandas as pd
import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import to_tensor, normalize
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import albumentations as A
import cv2
import PIL
import torchvision.transforms.v2 as v2
import torchvision.transforms as T
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
from torchvision.models.efficientnet import efficientnet_b0, efficientnet_b1

from TrainLoop.trainer import Trainer

np_dtype = np.float16
torch_dtype = torch.float16
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class DocDataset(Dataset):
    def __init__(self, img_root, data_csv_path, preprocessor):
        data_csv = pd.read_csv(data_csv_path)
        self.img_paths = [os.path.join(img_root, f) for f in data_csv['img_name'].values]
        self.target = data_csv['doc_type'].values
        self.preprocessor = preprocessor
        
    def __getitem__(self, idx):
        return self.preprocessor(Image.open(self.img_paths[idx])), self.target[idx]

    def __len__(self):
        return len(self.img_paths)


class ConvNextPreprocessor:
    def __init__(self, size, augmentator=None, use_imagenet_norm=True):
        def blank_aug(image): return {'image': image}
        
        if augmentator is not None:
            self.augmentator = augmentator
        else:
            self.augmentator = blank_aug
        if isinstance(self.augmentator, A.Compose):
            self.aug_backend = 'numpy'
        elif isinstance(self.augmentator, v2.Compose):
            self.aug_backend = 'torch'
        else:
            self.aug_backend = 'unspec'
        self.size = size
        # ¯\_(ツ)_/¯
        if use_imagenet_norm:
            self.mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).reshape(3,1,1)
            self.std = torch.tensor(IMAGENET_STD, dtype=torch.float32).reshape(3,1,1)
            self.get_mean = lambda x: self.mean
            self.get_std = lambda x: self.std
        else:
            self.get_mean = self._get_mean
            self.get_std = self._get_std
        
    def __call__(self, img: PIL.Image):
        img = self._preprocess_image(img)
        img = self.augment(img)
        return img
        
    def augment(self, img: torch.tensor):
        if self.aug_backend == 'numpy':
            transformed = self.augmentator(image=img.permute(1,2,0).numpy())
            img = transformed['image'].transpose(2,0,1)
        else:
            transformed = self.augmentator(image=img)
            img = transformed['image']
        return img
    
    def _preprocess_image(self, img):
        img = T.functional.to_tensor(img).unsqueeze(0)
        img = T.functional.resize(img, size=self.size, interpolation=T.InterpolationMode.BILINEAR, antialias=True).squeeze()
        img = (img - self.get_mean(img)) / self.get_std(img)
        return img
        
    @staticmethod
    def _get_mean(x: np.ndarray):
        return torch.mean(x, dim=(1,2)).reshape(3,1,1)
    
    @staticmethod
    def _get_std(x: np.ndarray):
        return torch.std(x, dim=(1,2)).reshape(3,1,1)
    
    
preprocessor_val = ConvNextPreprocessor(size, use_imagenet_norm=cfg['preprocessor']['use_imagenet_norm'], augmentator=None)

subset = 'Test'
ds_test = DocDataset(f'other/{subset}/images', f'other/doc_type_{subset}.csv', preprocessor_val)
dl_test = DataLoader(ds_test, batch_size=16, shuffle=False, num_workers=2)
