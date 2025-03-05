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

# class DocDataset(Dataset):
#     def __init__(self, img_root, data_csv_path, preprocessor):
#         data_csv = pd.read_csv(data_csv_path)
#         self.img_paths = [os.path.join(img_root, f) for f in data_csv['img_name'].values]
#         self.target = data_csv['doc_type'].values
#         self.preprocessor = preprocessor
        
#     def __getitem__(self, idx):
#         return self.preprocessor(Image.open(self.img_paths[idx])), self.target[idx]

#     def __len__(self):
#         return len(self.img_paths)
class DocDataset(Dataset):
    def __init__(self, img_root, preprocessor):
        self.img_paths = os.listdir(img_root)
        self.target = [1.0 if f.split('_')[-1].split('.')[0] == 'N' else 0.0 for f in self.img_paths]
        self.img_paths = [os.path.join(img_root, f) for f in self.img_paths]
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
        img[np.isnan(img)] = 0
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
        img = torch.concatenate([img, self.sobel(img)], axis=0)
        img = (img - self.get_mean(img)) / self.get_std(img)
        return img
        
    @staticmethod
    def _get_mean(x: np.ndarray):
        return torch.mean(x, dim=(1,2)).reshape(x.shape[0],1,1)
    
    @staticmethod
    def _get_std(x: np.ndarray):
        return torch.std(x, dim=(1,2)).reshape(x.shape[0],1,1)

    def sobel(self, img):
        scale = 1
        delta = 0
        ddepth = cv2.CV_16S

        gray = cv2.cvtColor(img.numpy().transpose(1,2,0), cv2.COLOR_BGR2GRAY)
        grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)

        grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)

        grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
        return torch.tensor(np.expand_dims(grad, axis=0))

class MetricAccum:
    def __init__(self):
        self.pred = []
        self.target = []
        
    def accum(self, p, t):
        self.pred.extend(p)
        self.target.extend(t)
        
    def compute(self):
        pred = np.array(self.pred)
        target = np.array(self.target)
        thr_range = np.arange(0.01, 0.99, 0.05)
        f1_values = []
        for thr in thr_range:
            f1_values.append(f1_score(target, pred > thr))
        best_thr = thr_range[np.argmax(f1_values)]
        
        res = {'f1': f1_score(target, pred > best_thr),
                'roc_auc': roc_auc_score(target, pred),
                'precision': precision_score(target, pred > best_thr),
                'recall': recall_score(target, pred > best_thr),
                'best_thr': best_thr}
        self._refresh_accumulations()
        return res
        
    def _refresh_accumulations(self):
        self.pred = []
        self.target = []

metric_accumulator = MetricAccum()

def accum_metrics_callback(preds, labels):
    preds = torch.sigmoid(preds).cpu().numpy()
    metric_accumulator.accum(preds.flatten().tolist(), labels.detach().cpu().numpy().tolist())
        
        
def compute_metrics_callback(curr_iter, train_hist, val_hist, save_dir, **state):
    res = metric_accumulator.compute()
    with open(os.path.join(save_dir, 'metrics.log'), 'a') as f:
            f.write(json.dumps(res, ensure_ascii=False) + '\n')
    
def save_train_progression_callback(train_hist, val_hist, lr_hist, eval_strat, save_dir, **state):
    os.makedirs(save_dir, exist_ok=True)
    f, ax = plt.subplots(1,3, figsize=(24,7))
    ax[0].plot(np.arange(len(train_hist)), train_hist)
    ax[0].set_title('Train Loss')
    ax[0].set_xlabel(f'Iterations({eval_strat})')
    ax[0].grid()
    ax[1].plot(np.arange(len(val_hist)), val_hist)
    ax[1].set_title('val Loss')
    ax[1].set_xlabel(f'Iterations({eval_strat})')
    ax[1].grid()
    ax[2].plot(np.arange(len(lr_hist)), lr_hist)
    ax[2].set_title('Learning Rate')
    ax[2].set_xlabel(f'Iterations({eval_strat})')
    ax[2].grid()
    plt.savefig(os.path.join(save_dir, 'train_plots.jpg'))
    plt.clf()
    plt.close()
    

optimizer_mapping = {'AdamW': torch.optim.AdamW}
scheduler_mapping = {'ExponentialLR': torch.optim.lr_scheduler.ExponentialLR,
                     'CosineAnnealing': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts}

def save_train_stats(train_hist, val_hist, lr_hist, eval_strat, save_path):
    joblib.dump({'train_hist': train_hist,
                 'val_hist': val_hist,
                 'lr_hist': lr_hist,
                 'eval_strat': eval_strat}, os.path.join(save_path, 'train_stats.pickle'))

def save_random_train_imgs(ds, save_path):
    os.makedirs(save_path, exist_ok=True)
    f, ax = plt.subplots(4,4, figsize=(12,7))
    random_idxs = np.random.choice(np.arange(len(ds)), 16, replace=False)
    for i, idx in enumerate(random_idxs):
        ax[i//4, i%4].imshow(np.array(ds[idx][0]).transpose(1,2,0))
        ax[i//4, i%4].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'train_img_examples.jpg'))
    plt.clf()
    plt.close()

    
def main(args):
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    

    # ---------- Data settings ----------
    size = cfg['preprocessor']['size']
    augmentations = A.Compose([
        A.Perspective(scale=cfg['aug']['perspective']['scale'], keep_size=True),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        # A.GaussianBlur(blur_limit=cfg['aug']['blur']['blur_limit'], sigma_limit=cfg['aug']['blur']['sigma_limit'], p=cfg['aug']['blur']['p']),
        # A.RandomBrightnessContrast(p=cfg['aug']['brightness_contrast']['p']),
        # A.ColorJitter(cfg['aug']['jitter']['p']),
        # A.Emboss(alpha=cfg['aug']['emboss']['alpha'], strength=cfg['aug']['emboss']['str'], p=cfg['aug']['emboss']['p']),
        # A.ToGray(cfg['aug']['gray']['p']),
        A.Rotate(limit=[*cfg['aug']['rotate']['angles']]),
        
    ])

    preprocessor_train = ConvNextPreprocessor(size, use_imagenet_norm=cfg['preprocessor']['use_imagenet_norm'], augmentator=augmentations)
    preprocessor_val = ConvNextPreprocessor(size, use_imagenet_norm=cfg['preprocessor']['use_imagenet_norm'], augmentator=None)
    
    subset = 'Train'
    ds_train = DocDataset(f'readability_clf/{subset}/', preprocessor_train)
    subset = 'Validation'
    ds_val = DocDataset(f'readability_clf/{subset}/', preprocessor_val)
    subset = 'Test'
    ds_test = DocDataset(f'readability_clf/{subset}/', preprocessor_val)
    
    save_random_train_imgs(ds_train, os.path.join('readability_clf/runs', cfg['run_name']))
    
    dl_train = DataLoader(ds_train, batch_size=cfg['dataset']['batch_size'], shuffle=True, num_workers=cfg['dataset']['workers'])
    dl_val = DataLoader(ds_val, batch_size=cfg['dataset']['batch_size'], shuffle=False, num_workers=cfg['dataset']['workers'])
    
    
    # ---------- Create Model ----------
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device:', device)
    # model = timm.create_model(cfg['convnext_model'])
    # model.head.fc = nn.Linear(in_features=384, out_features=1, bias=True)
    model = efficientnet_b0()
    # model.load_state_dict(torch.load('models/efficientnet_b0.pth'))
    model.features[0][0] = torch.nn.Conv2d(4, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 1)
    
    optimizer = optimizer_mapping[cfg['train']['optimizer']['name']](model.parameters(), **cfg['train']['optimizer']['params'])
    scheduler = scheduler_mapping[cfg['train']['scheduler']['name']](optimizer, **cfg['train']['scheduler']['params'])
    
    bce_loss = nn.BCEWithLogitsLoss()
    def loss(pred, target):
        return bce_loss(pred, target.reshape(-1,1).type(torch.float32))
    
    
    # ---------- Trainer ----------
    trainer = Trainer(model=model,
            save_dir=os.path.join('readability_clf/runs', cfg['run_name']),
            loss_fn=loss,
            optimizer=optimizer,
            scheduler=scheduler,
            n_epochs=cfg['train']['n_epochs'],
            device=device,
            early_stop_rounds=cfg['train']['early_stop_rounds'],
            eval_strat=cfg['train']['eval_strat'],
            on_batch_validation_callbacks=[accum_metrics_callback],
            end_of_iter_callbacks=[compute_metrics_callback, save_train_progression_callback],
            save_only_best=cfg['train']['save_only_best'])
    
    
    # ---------- Run ----------
    trainer.train(dl_train, dl_val)
    
    save_train_stats(trainer.train_hist, trainer.val_hist, trainer.lr_hist, trainer.eval_strat, os.path.join('readability_clf/runs', cfg['run_name']))
                
    with open(os.path.join('readability_clf/runs', cfg['run_name'], 'config.yaml'), 'w') as f:
        yaml.safe_dump(cfg, f)
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    
    args = parser.parse_args()
    
    main(args)