import os

from torch.utils.data import Dataset
from pycocotools.coco import COCO
import torchvision.transforms as T
import numpy as np
import torch
import PIL
from PIL import img
import albumentations as A
import torchvision.transforms.v2 as v2


class InstanceSegCOCODataset(Dataset):
    def __init__(self, img_path, mask_path, coco_path, preprocessor):
        self.img_path = img_path
        self.mask_path = mask_path
        self.coco_dataset = COCO(coco_path)
        self.cat_ids = self.coco_dataset.getCatIds()
        self.idx_mapper = {i: coco_img_id for i, coco_img_id in enumerate(sorted(self.coco_dataset.imgs.keys()))}
        self.preprocesspr = preprocessor

    def __getitem__(self, idx):
        coco_img_id = self.idx_mapper[idx]
        coco_img = self.coco_dataset.imgs[coco_img_id]
        img = img.open(os.path.join(self.img_path, coco_img['file_name']))
        if img.mode != 'RGB':
            img = img.convert('RGB')

        ann_ids = self.coco_dataset.getAnnIds(imgIds=coco_img_id, catIds=self.cat_ids, iscrowd=None)
        anns = self.coco_dataset.loadAnns(ann_ids)

        mask = np.zeros((coco_img['height'], coco_img['width']), dtype=np.int32)
        for ann in anns:
            ann_mask = self.coco_dataset.annToMask(ann)
            mask[ann_mask == 1] = ann['category_id']

        self.preprocessor(img, mask)

    
    def __len__(self):
        return len(self.coco_dataset.imgs)
    

class InstanceSegDETR(Dataset):
    """
    Dataset for DETRForSegmentation.
    Calls the DETR feature extractor inside __getitem__.
    """
    def __init__(self, img_path, coco_path, preprocessor):
        self.img_path = img_path
        self.coco_dataset = COCO(coco_path)
        self.cat_ids = self.coco_dataset.getCatIds()
        self.idx_mapper = {i: coco_img_id for i, coco_img_id in enumerate(sorted(self.coco_dataset.imgs.keys()))}
        self.preprocessor = preprocessor

    def __getitem__(self, idx):
        coco_img_id = self.idx_mapper[idx]
        coco_img = self.coco_dataset.imgs[coco_img_id]
        image = PIL.Image.open(os.path.join(self.img_path, coco_img['file_name']))
        if image.mode != 'RGB':
            image = image.convert('RGB')

        ann_ids = self.coco_dataset.getAnnIds(imgIds=coco_img_id, catIds=self.cat_ids, iscrowd=None)
        anns = self.coco_dataset.loadAnns(ann_ids)

        # Build instance_map: each instance gets a unique id (0=background)
        segments_info = []
        instance_id = 1
        instance_map = np.zeros((coco_img['height'], coco_img['width']), dtype=np.int32)
        for ann in anns:
            ann_mask = self.coco_dataset.annToMask(ann)
            instance_map[ann_mask == 1] = instance_id
            instance_id += 1

            segments_info.append({
                'id': instance_id,
                'category_id': ann['category_id'],
                'bbox': ann['bbox'],
                'iscrowd': ann.get('iscrowd', 0),
                'area': ann.get('area', 0)
            })
            instance_id += 1

        target = {
            "image_id": coco_img_id,
            "segments_info": segments_info,
            "height": coco_img['height'],
            "width": coco_img['width'],
        }

        # Call the DETR feature extractor
        features = self.feature_extractor(
            image,
            annotations=anns,
            masks=instance_map,
            return_tensors="pt"
        )
        # Короче маски бинарные размера M где M это кол-во инстанс объектов 
        # model.detr.model.backbone (что является классом DetrConvModel) поменять на другое свое
        return features

    def __len__(self):
        return len(self.coco_dataset.imgs)
