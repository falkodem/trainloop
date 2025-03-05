import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from archs.unet import UNetConvNext, UNetEffNet
from archs.convnext_seg import ConvNextSeg
from TrainLoop.data import ConvNextPreprocessor, SemanticSegmentationCOCODataset, CarSegmentationDataset

def infer_and_visualize(model, ds, id2label, id2color, i_img, thr, reduced_labels):
    img_path = os.path.join(ds.img_dir, ds.coco_dataset.imgs[ds.idx_mapper[i_img]]['file_name'])
    # img_path = ds.img_files[i_img]
    orig_img = cv2.imread(img_path)[:,:,::-1]
    orig_img = cv2.resize(orig_img, (ds.preprocessor.size[1], ds.preprocessor.size[0]))
    
    gt = ds[i_img][1].detach().numpy().astype(np.uint8)
    print('GT labels:', [(idx, id2label[idx]) for idx in np.unique(gt)])
    x = ds[i_img][0].unsqueeze(0)
    with torch.no_grad():
        model.eval()
        outputs = model(x)
    logits = outputs.detach().cpu()
    probs = torch.softmax(nn.functional.interpolate(logits,
                    size=orig_img.shape[:2], # (height, width)
                    mode='bilinear',
                    align_corners=False)[0], axis=0).numpy()
    # probs = torch.sigmoid(nn.functional.interpolate(logits,
    #                 size=orig_img.shape[:2], # (height, width)
    #                 mode='bilinear',
    #                 align_corners=False)[0]).numpy()
    if reduced_labels:
        pred_label = np.argmax(np.concatenate([np.zeros((1, *probs.shape[1:]), dtype=np.float32) + thr,
                                               np.where(probs > thr, probs, 0)], axis=0), axis=0)
    else:
        pred_label = np.argmax(probs, axis=0)
        # pred_label = np.argmax(np.where(probs>thr, probs, 0), axis=0)
    print('Pred labels:', [(idx, id2label[idx]) for idx in np.unique(pred_label)])

    f, ax = plt.subplots(1,3, figsize=(20,7))
    color_seg = np.zeros((gt.shape[0], gt.shape[1], 3), dtype=np.uint8) # height, width, 3
    for label, color in id2color.items():
        color_seg[gt == label, :] = color
    ax[0].imshow((color_seg*0.7 + orig_img*0.3).astype(np.uint8))
    
    color_seg = np.zeros((pred_label.shape[0], pred_label.shape[1], 3), dtype=np.uint8) # height, width, 3
    for label, color in id2color.items():
        color_seg[pred_label == label, :] = color
        print(label)
    ax[1].imshow((color_seg*0.7 + orig_img*0.3).astype(np.uint8))
    ax[2].imshow(orig_img)
    plt.savefig(f'tmp/visualization{i_img}.jpg')

    # f, ax = plt.subplots(1,3, figsize=(20,7))
    # color_seg = np.repeat(np.expand_dims(gt, axis=2), 3, axis=2)*id2color[0]
    # ax[0].imshow((color_seg*0.7 + orig_img*0.3).astype(np.uint8))
    # color_seg = np.zeros((pred_label.shape[0], pred_label.shape[1], 3), dtype=np.uint8) # height, width, 3
    # for label, color in id2color.items():
    #     color_seg[pred_label == label, :] = color
    # ax[1].imshow((color_seg*0.7 + orig_img*0.3).astype(np.uint8))
    # ax[2].imshow(orig_img)
    # plt.savefig(f'tmp/visualization{i_img}.jpg')



model = torch.load('./runs/damages_unet_effnet//best.pt', map_location='cpu')
# model = torch.load('./runs/car_segment_dicefocal_effnet_testval_swap/best.pt', map_location='cpu')
# model = torch.load('./runs/car_segment_effnet_focaldice/best.pt', map_location='cpu')

model.eval()

size = (768,768)
preprocessor = ConvNextPreprocessor(size, use_imagenet_norm=False)
subset = 'Test'
ds_test = SemanticSegmentationCOCODataset(f'../segformer/datasets/damages_1k_v3_enlarge1_reshuffle_coco/{subset}',
                                          f'../segformer/datasets/damages_1k_v3_enlarge1_reshuffle_coco/{subset}/instances_{subset}.json', preprocessor)
# ds_test = CarSegmentationDataset(f'./datasets/car_mask_from_elements/{subset}/images',
                                      # f'./datasets/car_mask_from_elements/{subset}/anns', preprocessor)


id2label = {v['id']: v['name'] for _,v in ds_test.coco_dataset.cats.items()}
id2label[0] = 'фон'
label2id = {name: id_ for id_, name in id2label.items()}
id2color = {0: [0, 0, 0], 1: [200, 0, 0], 2: [108, 64, 20], 3: [255, 229, 204], 4: [0, 102, 0], 5: [0, 255, 0], 6: [0, 153, 153], 7: [0, 128, 255], 8: [255, 255, 0]}

# id2label = {0: 'фон', 1: 'car'}
# label2id = {name: id_ for id_, name in id2label.items()}
# id2color = {0: [0,0,0], 1:[255, 0, 0]}

reduced_labels=False


for i_img in range(1):
    infer_and_visualize(model=model, ds=ds_test, id2label=id2label, 
                        id2color=id2color, i_img=i_img, thr=0.50,
                        reduced_labels=False)