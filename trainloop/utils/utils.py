import os

import torch
import joblib
import matplotlib.pyplot as plt
import numpy as np

def one_hot_labels(labels: torch.Tensor, n_classes, dim=1):
    '''
    Params:
        labels (torch.Tensor): BxCxHxW, where C - channels and should be equal to 1
    '''
    sh = list(labels.shape)
    if sh[dim] != 1:
        raise AssertionError(f'Labels dim (for that case dim{dim}) should be equal to 1 (now is equal to {sh[dim]})')
    sh[dim] = n_classes
    return torch.zeros(size=sh, dtype=labels.dtype, device=labels.device).scatter_(dim=dim, index=labels, value=1)


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
        data = ds[idx]
        img = np.array(data[0]).transpose(1,2,0)
        img = img + np.abs(img.min(axis=(0,1)))
        img = img * (255 / img.max(axis=(0,1)))
        ann = np.array(data[1])
        ann = np.repeat(ann[..., None], 3, axis=2)
        for cls in np.unique(ann):
            if cls == 0:
                continue
            color = np.random.randint(0, 255, 3)
            ann[ann[:,:,0] == cls] = color
        ann = ann.astype(np.uint8)
        res = (img*0.3 + ann*0.7).astype(np.uint8)
        ax[i//4, i%4].imshow(res)
        ax[i//4, i%4].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'train_img_examples.jpg'))
    plt.clf()
    plt.close()


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