import os
import json
import argparse
import yaml
import joblib

import timm
from torchvision.models.efficientnet import efficientnet_b0
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_tensor
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import albumentations as A
import cv2

from TrainLoop.trainer import Trainer
from archs.unet import UNetConvNext, UNetEffNet
from archs.convnext_seg import ConvNextSeg
from TrainLoop.data import SemanticSegmentationCOCODataset, ConvNextPreprocessor, ConvNextPreprocessorNumpy, CarSegmentationDataset
from TrainLoop.utils import one_hot_labels
from TrainLoop.metrics import MeanIOU
from TrainLoop.loss import DiceLoss

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
        img = np.array(ds[idx][0]).transpose(1,2,0)
        img = img + np.abs(img.min(axis=(0,1)))
        img = img * (255 / img.max(axis=(0,1)))
        ann = np.array(ds[idx][1])
        ann = ann * (255 / ann.max())
        res = (img*0.3).astype(np.uint8) + (np.repeat(np.expand_dims(ann, axis=2), 3, axis=2)*0.7).astype(np.uint8)
        ax[i//4, i%4].imshow(res)
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
        A.CropNonEmptyMaskIfExists(width=cfg['aug']['crop_size'][0], height=cfg['aug']['crop_size'][1]),
        A.Perspective(scale=[*cfg['aug']['perspective']['scale']], keep_size=True),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=cfg['aug']['brightness_contrast']['p']),
        A.Rotate(limit=[*cfg['aug']['rotate']['angles']]),
        A.Resize(height=size[0], width=size[1])
    ])

    preprocessor_train = ConvNextPreprocessor(size, use_imagenet_norm=cfg['preprocessor']['use_imagenet_norm'], augmentator=augmentations)
    preprocessor_val = ConvNextPreprocessor(size, use_imagenet_norm=cfg['preprocessor']['use_imagenet_norm'])
    
    # subset = 'Train'
    # ds_train = CarSegmentationDataset(f'./datasets/car_mask_from_elements/{subset}/images', f'./datasets/car_mask_from_elements/{subset}/anns', preprocessor_train)
    # subset = 'Validation'
    # ds_val = CarSegmentationDataset(f'./datasets/car_mask_from_elements/{subset}/images', f'./datasets/car_mask_from_elements/{subset}/anns', preprocessor_val)
    # subset = 'Test'
    # ds_test = CarSegmentationDataset(f'./datasets/car_mask_from_elements/{subset}/images', f'./datasets/car_mask_from_elements/{subset}/anns', preprocessor_val)

    subset = 'train'
    ds_train = SemanticSegmentationCOCODataset(f'../data/car_damage_small/{subset}',
                                               f'../data/car_damage_small/{subset}/_annotations.coco.json',
                                               preprocessor_train)
    subset = 'valid'
    ds_val = SemanticSegmentationCOCODataset(f'../data/car_damage_small/{subset}',
                                             f'../data/car_damage_small/{subset}/_annotations.coco.json',
                                             preprocessor_val)
    subset = 'test'
    ds_test = SemanticSegmentationCOCODataset(f'../data/car_damage_small/{subset}',
                                              f'../data/car_damage_small/{subset}/_annotations.coco.json',
                                              preprocessor_val)
    
    save_random_train_imgs(ds_train, os.path.join('runs', cfg['run_name']))

    dl_train = DataLoader(ds_train, batch_size=cfg['dataset']['batch_size'], shuffle=True, num_workers=cfg['dataset']['workers'])
    dl_val = DataLoader(ds_val, batch_size=cfg['dataset']['batch_size'], shuffle=False, num_workers=cfg['dataset']['workers'])

    id2label = {v['id']: v['name'] for _,v in ds_train.coco_dataset.cats.items()}
    id2label[0] = 'фон'
    label2id = {name: id_ for id_, name in id2label.items()}
    # id2label = {0: 'фон', 1: 'car'}
    # label2id = {name: id_ for id_, name in id2label.items()}
    
        
    
    # ---------- Create Model ----------
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device:', device)
    convnext = timm.create_model(cfg['convnext_model'], checkpoint_path = cfg['convnext_model_path'])
    # model = efficientnet_b0()
    model = UNetConvNext(convnext, n_classes=len(label2id))
    # model = ConvNextSeg(convnext, n_classes=len(label2id))
    # model = UNetEffNet(model, n_classes=len(label2id))
    
    optimizer = optimizer_mapping[cfg['train']['optimizer']['name']](model.parameters(), **cfg['train']['optimizer']['params'])
    scheduler = scheduler_mapping[cfg['train']['scheduler']['name']](optimizer, **cfg['train']['scheduler']['params'])

    
    
    # ---------- Loss settings ----------
    
    # ----- CE -----
    # def ce_loss(preds, labels):
    #     preds = nn.functional.interpolate(preds,
    #             size=labels.shape[1:], # (height, width),
    #             mode='bilinear',
    #             align_corners=False)
    #     labels = labels.type(torch.LongTensor).to(device)
    #     return torch.nn.functional.cross_entropy(preds, labels)
    
    # ----- DICE -----
    # dice_loss_fn = DiceLoss(**cfg['train']['loss']['dice'])
    # def dice_loss(preds, labels):
    #     # preds = nn.functional.interpolate(preds,
    #     #         size=labels.shape[1:], # (height, width),
    #     #         mode='bilinear',
    #     #         align_corners=False)
    #     # labels = labels.unsqueeze(1).type(torch.int64)
    #     # labels = one_hot_labels(labels, n_classes=len(id2label), dim=1)
    #     # preds = torch.softmax(preds, dim=1)
    #     # return dice_loss_fn(preds, labels)

    #     labels = nn.functional.interpolate(labels.unsqueeze(1).to(torch.float32),
    #         size=preds.shape[2:], # (height, width)
    #         mode='nearest').to(torch.int64)
    #     labels = one_hot_labels(labels, n_classes=len(id2label), dim=1)
    #     preds = torch.softmax(preds, dim=1)
    #     return dice_loss_fn(preds, labels)
                
    # from segmentation_models_pytorch.losses import DiceLoss
    # dice_loss_fn = DiceLoss(mode='multiclass')
    # def dice_loss(preds, labels):
    #     preds = nn.functional.interpolate(preds,
    #             size=labels.shape[1:], # (height, width),
    #             mode='bilinear',
    #             align_corners=False)
    #     labels = labels.unsqueeze(1).type(torch.int64)
    #     return dice_loss_fn(y_pred=preds, y_true=labels)
    
    # ----- DICE + Focal -----
    from segmentation_models_pytorch.losses import DiceLoss, FocalLoss
    dice_loss_fn = DiceLoss(mode='multiclass', ignore_index=0)
    focal_loss_fn = FocalLoss(mode='multiclass')
    def dice_focal_loss(preds, labels):
        preds = nn.functional.interpolate(preds,
                size=labels.shape[1:], # (height, width),
                mode='bilinear',
                align_corners=False)
        labels = labels.unsqueeze(1).type(torch.int64)

        # labels = nn.functional.interpolate(labels.unsqueeze(1).to(torch.float32),
        #     size=preds.shape[2:], # (height, width)
        #     mode='nearest').to(torch.int64)

        return dice_loss_fn(y_pred=preds, y_true=labels) + focal_loss_fn(y_pred=preds, y_true=labels.squeeze(1))
    
    
    # ---------- Evaluator settings ----------
    mean_iou = MeanIOU(num_labels=len(id2label), mode='accum', reduce_labels=False)
    
    def accum_iou_callback(preds, labels):
        # preds = nn.functional.interpolate(preds.detach(),
        #         size=labels.shape[1:], # (height, width),
        #         mode='bilinear',
        #         align_corners=False)
        # preds = np.argmax(torch.softmax(preds, dim=1).cpu().numpy(), axis=1)
        # mean_iou.compute(preds, labels.cpu())
        
        preds = np.argmax(torch.softmax(preds, dim=1).cpu().numpy(), axis=1)
        labels = nn.functional.interpolate(labels.unsqueeze(1).type(torch.FloatTensor),
                    size=preds.shape[1:], # (height, width)
                    mode='nearest').squeeze(1).numpy()
        mean_iou.compute(preds, labels)

    def compute_iou_callback(curr_iter, train_hist, val_hist, save_dir, **state):
        res = mean_iou.get_results()
        for i_label, iou in enumerate(res['per_category_iou']):
            res[f'{id2label[i_label]}_iou'] = iou
        res.pop('per_category_iou')
        for i_label, acc in enumerate(res['per_category_accuracy']):
            res[f'{id2label[i_label]}_acc'] = acc
        res.pop('per_category_accuracy')
        res['iteration'] = curr_iter
        res['train_loss'] = train_hist[-1]
        res['val_loss'] = val_hist[-1]
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
        
        
    # ---------- Trainer ----------
    trainer = Trainer(model=model,
            save_dir=os.path.join('runs', cfg['run_name']),
            loss_fn=dice_focal_loss,
            optimizer=optimizer,
            scheduler=scheduler,
            n_epochs=cfg['train']['n_epochs'],
            device=device,
            early_stop_rounds=cfg['train']['early_stop_rounds'],
            eval_strat=cfg['train']['eval_strat'],
            on_batch_validation_callbacks=[accum_iou_callback],
            end_of_iter_callbacks=[compute_iou_callback, save_train_progression_callback],
            save_only_best=cfg['train']['save_only_best'])
    
    
    # ---------- Run ----------
    trainer.train(dl_train, dl_val)
    
    save_train_stats(trainer.train_hist, trainer.val_hist, trainer.lr_hist, trainer.eval_strat, os.path.join('runs', cfg['run_name']))
                
    with open(os.path.join('runs', cfg['run_name'], 'config.yaml'), 'w') as f:
        yaml.safe_dump(cfg, f)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    
    args = parser.parse_args()
    
    main(args)


