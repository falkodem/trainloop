import os
import json

import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_tensor
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import albumentations as A

from TrainLoop.trainer import Trainer
from archs.unet import UNetConvNext
from TrainLoop.data import SemanticSegmentationCOCODataset, ConvNextPreprocessor, ConvNextPreprocessorNumpy
from TrainLoop.utils import one_hot_labels
from TrainLoop.metrics import MeanIOU
from TrainLoop.loss import DiceLoss

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device:',device)
convnext = timm.create_model('convnextv2_femto', checkpoint_path = './models/convnextv2_femto.safetensors')

augmentations = A.Compose([
    A.RandomCrop(width=512, height=512),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
])

size = (768,768)
preprocessor = ConvNextPreprocessor(size, use_imagenet_norm=False)

subset = 'train'
ds_train = SemanticSegmentationCOCODataset(f'../data/car_damage_small/{subset}','_annotations.coco.json', preprocessor)
subset = 'valid'
ds_val = SemanticSegmentationCOCODataset(f'../data/car_damage_small/{subset}','_annotations.coco.json', preprocessor)
subset = 'test'
ds_test = SemanticSegmentationCOCODataset(f'../data/car_damage_small//{subset}','_annotations.coco.json', preprocessor)

dl_train = DataLoader(ds_train, batch_size=8, shuffle=True, num_workers=4)
dl_val = DataLoader(ds_val, batch_size=8, shuffle=False, num_workers=4)

id2label = {v['id']: v['name'] for _,v in ds_train.coco_dataset.cats.items()}
id2label[0] = 'фон'
label2id = {name: id_ for id_, name in id2label.items()}

n_epochs = 40

unet = UNetConvNext(convnext, n_classes=len(label2id))
optimizer = torch.optim.AdamW(unet.parameters(), lr=3e-4)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

def ce_loss(preds, labels):
    labels = labels.unsqueeze(1)
    labels = nn.functional.interpolate(labels.type(torch.FloatTensor),
            size=preds.shape[2:], # (height, width)
            mode='bilinear',
            align_corners=False).type(torch.LongTensor).to(device).squeeze(1)
    return torch.nn.functional.cross_entropy(preds, labels)

dice_loss_fn = DiceLoss(include_background=False, squared_pred=False, jaccard=False, reduction='mean', weight=None)
def dice_loss(preds, labels):
    labels = labels.unsqueeze(1)
    labels = nn.functional.interpolate(labels.type(torch.FloatTensor),
            size=preds.shape[2:], # (height, width)
            mode='bilinear',
            align_corners=False).type(torch.LongTensor).to(device)
    labels = one_hot_labels(labels, n_classes=len(id2label), dim=1)
    preds = torch.softmax(preds, dim=1)
    return dice_loss_fn(preds, labels)

mean_iou = MeanIOU(num_labels=len(id2label), mode='accum')

def accum_iou_callback(preds, labels):
    preds = np.argmax(torch.softmax(preds.cpu(), dim=1).numpy(), axis=1)
    labels = nn.functional.interpolate(labels.unsqueeze(1).type(torch.FloatTensor),
                size=preds.shape[1:], # (height, width)
                mode='nearest').squeeze(1).numpy()
    mean_iou.compute(preds, labels)

def compute_iou_callback(**state):
    res = mean_iou.get_results()
    for i_label, iou in enumerate(res['per_category_iou']):
        res[f'{id2label[i_label]}_iou'] = iou
    res.pop('per_category_iou')
    for i_label, acc in enumerate(res['per_category_accuracy']):
        res[f'{id2label[i_label]}_acc'] = acc
    res.pop('per_category_accuracy')
    res['iteration'] = state['curr_iter']
    res['train_loss'] = state['train_loss']
    res['val_loss'] = state['val_loss']   
    with open(os.path.join(state['save_dir'],'metrics.log'), 'a') as f:
            f.write(json.dumps(res, ensure_ascii=False) + '\n')

trainer = Trainer(model=unet,
        save_dir='run/base_dice_noBack',
        loss_fn=dice_loss,
        optimizer=optimizer,
        scheduler=scheduler,
        n_epochs=40,
        device=device,
        early_stop_rounds=5,
        eval_strat='epoch',
        accumulate_metric_callbacks=[accum_iou_callback],
        compute_metric_callbacks=[compute_iou_callback],
        save_only_best=True)

trainer.train(dl_train, dl_val)