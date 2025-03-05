import joblib
import os
import json

import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2

from TrainLoop.trainer import Trainer
# from archs.unet import UNetConvNext
# from archs.convnext_seg import ConvNextSeg
# from TrainLoop.data import SemanticSegmentationCOCODataset, ConvNextPreprocessor, ConvNextPreprocessorNumpy
# from TrainLoop.utils import one_hot_labels
# from TrainLoop.metrics import MeanIOU
# from TrainLoop.loss import DiceLoss


SEED = 42

class AdversarialDataset(Dataset):
    def __init__(self, img_files, target, size):
        self.image_files = img_files
        self.target = target
        self.size = size
            
    def __getitem__(self, idx):
        return self._preprocess_image(Image.open(self.image_files[idx])), self.target[idx]
    
    def __len__(self):
        return len(self.target)
    
    def _preprocess_image(self, img):
        img = T.functional.to_tensor(img).unsqueeze(0)
        img = T.functional.resize(img, size=self.size, interpolation=T.InterpolationMode.BILINEAR, antialias=True).squeeze()
        return img
    

train_root = f'../segformer/datasets/damages_1k_v3_enlarge1_reshuffle_coco/Train'
test_root = f'../segformer/datasets/damages_1k_v3_enlarge1_reshuffle_coco/Validation'

train_files = []
for root, d, files in os.walk(train_root):
    train_files.extend([os.path.join(root, f) for f in files if 'json' not in f])
test_files = []
for root, d, files in os.walk(test_root):
    test_files.extend([os.path.join(root, f) for f in files if 'json' not in f])

image_files = train_files + test_files
target = [0 for _ in range(len(train_files))] + [1 for _ in range(len(test_files))]

img_train, img_test, target_train, target_test = train_test_split(image_files, target, test_size=0.2, shuffle=True)

ds_train = AdversarialDataset(img_train, target_train, (640, 640))
ds_test = AdversarialDataset(img_test, target_test, (640, 640))

# model = timm.create_model('convnextv2_femto', checkpoint_path = './models/convnextv2_femto.safetensors')
# model.head.fc = torch.nn.Linear(384, 1)

# loss_fn = torch.nn.BCEWithLogitsLoss()

# def loss(preds, target):
#     return loss_fn(preds, target.reshape(-1,1).type(torch.float32))

# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
# device = 'cuda'

# test_preds = []
# test_gts = []
# thr = 0.5
        
# def accum_stats(preds, labels):
#     preds = torch.sigmoid(preds.detach().cpu())
#     test_preds.extend(preds.tolist())
#     test_gts.extend(labels.detach().cpu().numpy().tolist())
    
# def compute_metric(save_dir, **state):
#     global test_preds, test_gts
#     res = {}
#     res['f1'] = f1_score(test_gts, np.array(test_preds)>thr)
#     res['roc_auc'] = roc_auc_score(test_gts, test_preds)
#     res['prec'] = precision_score(test_gts, np.array(test_preds)>thr)
#     res['rec'] = recall_score(test_gts, np.array(test_preds)>thr)
#     with open(os.path.join(save_dir, 'metrics.log'), 'a') as f:
#         f.write(json.dumps(res, ensure_ascii=False) + '\n')
#     test_preds, test_gts = [], []
    

# trainer = Trainer(model=model,
#             save_dir=os.path.join('runs', 'adversarial'),
#             loss_fn=loss,
#             optimizer=optimizer,
#             scheduler=None,
#             n_epochs=50,
#             device=device,
#             early_stop_rounds=10,
#             eval_strat='epoch',
#             on_batch_validation_callbacks=[accum_stats],
#             end_of_iter_callbacks=[compute_metric],
#             save_only_best=True)


# dl_train = DataLoader(ds_train, batch_size=16, shuffle=True)
# dl_test = DataLoader(ds_test, batch_size=16, shuffle=False)
# trainer.train(dl_train, dl_test)


from tqdm import tqdm
model = torch.load('./runs/adversarial/best.pt')
with torch.no_grad():
    preds = []
    for x, y in ds_test:
        x = x.to('cuda')
        preds.append(model(x.unsqueeze(0)).detach().cpu().numpy().item())
        
import matplotlib.pyplot as plt
plt.hist(preds)
plt.savefig('PREDS.jpg')
idxs = np.argsort(preds)[-10:]
print(np.array(preds)[idxs])
for i in idxs:
    print(ds_test.image_files[i])