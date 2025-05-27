import os
from typing import Union

from torch.utils.data import Dataset
from pycocotools.coco import COCO
import torchvision.transforms as T
import torch.nn.functional as F
import numpy as np
import torch
import PIL
from PIL import Image
import albumentations as A
import torchvision.transforms.v2 as v2

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

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
        img = Image.open(os.path.join(self.img_path, coco_img['file_name']))
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
    InstanceSegDETR is a PyTorch Dataset class for instance segmentation tasks using DETRForSegmentation.

    This dataset loads images and their corresponding instance segmentation annotations from a COCO-style dataset.
    For each image, it creates a binary mask for each instance and collects relevant annotation information such as
    class labels, bounding boxes, iscrowd flags, and area. The dataset applies a preprocessing function to both the
    image and the target dictionary, which can include resizing, normalization, and augmentations.

    Args:
        img_path (str): Path to the directory containing images.
        coco_path (str): Path to the COCO annotation file (JSON).
        preprocessor (callable): A function or callable object that processes the image and target.

    Returns:
        A tuple (image, target) where:
            - image: The preprocessed image tensor.
            - target: A dictionary containing:
                - "image_id": Image ID from COCO.
                - "masks": Instance binary masks (n_instances x H x W).
                - "class_labels": Array of category IDs for each instance.
                - "size": [height, width] of the image.
                - "orig_size": Original [height, width] of the image.
                - "boxes": Bounding boxes for each instance.
                - "iscrowd": iscrowd flags for each instance.
                - "area": Area for each instance.
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
        # Create a binary mask with shape: n_instances x height x width. Fill each channel with binary mask
        instance_map = np.zeros((len(anns), coco_img['height'], coco_img['width']), dtype=np.uint8)
        for i_instance, ann in enumerate(anns):
            ann_mask = self.coco_dataset.annToMask(ann)
            instance_map[i_instance] = ann_mask

        target = {
            "image_id": coco_img_id,
            "masks": instance_map,
            "class_labels": np.asarray([ann['category_id'] for ann in anns], dtype=np.int64),
            "size": np.asarray([coco_img['height'], coco_img['width']], dtype=np.int64),
            "orig_size": np.asarray([coco_img['height'], coco_img['width']], dtype=np.int64),
            "boxes": np.asarray([ann['bbox'] for ann in anns], dtype=np.float32),
            "iscrowd": np.asarray([ann.get('iscrowd', 0) for ann in anns], dtype=np.int64),
            "area": np.asarray([ann.get('area', 0) for ann in anns], dtype=np.float32),
        }
        # Here is resize, augmentations, etc.
        data = self.preprocessor(image, target)

        # model.detr.model.backbone (что является классом DetrConvModel) поменять на другое свое
        return data

    def __len__(self):
        return len(self.coco_dataset.imgs)


class DetrPreprocessor:
    def __init__(self, size, augmentator=None, use_imagenet_norm=True):
        self.size = size
        self.augmentator = augmentator
        self.use_imagenet_norm = use_imagenet_norm
        if use_imagenet_norm:
            self._imagenet_mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).reshape(3,1,1)
            self._imagenet_std = torch.tensor(IMAGENET_STD, dtype=torch.float32).reshape(3,1,1)
        if isinstance(self.augmentator, A.Compose):
            self.aug_backend = 'numpy'
        else:
            self.aug_backend = 'torch'

    def __call__(self, image: PIL.Image, target: Union[dict, None]=None):
        image = T.functional.to_tensor(image)
        if target is not None:
            target['masks'] = torch.tensor(target['masks'], dtype=torch.uint8)
            target['boxes'] = torch.tensor(target['boxes'], dtype=torch.float32)
        else:
            target = None
        # Resize with padding, augment
        image, mask, boxes = self.resize_keeping_aspect_ratio(image,
                                            target['masks'] if target else target,
                                            target['boxes'] if target else target)
        image = (image - self._get_mean(image)) / self._get_std(image)
        image, pixel_mask, mask = self.pad_to_required_size(image, mask)
        
        if self.augmentator:
            image, pixel_mask_and_seg_mask, boxes = self.augment(image,
                                                        torch.concatenate([pixel_mask.unsqueeze(0), mask]),
                                                        boxes)
            pixel_mask, mask = pixel_mask_and_seg_mask[0], pixel_mask_and_seg_mask[1:]
            del pixel_mask_and_seg_mask

        processed_data = {}
        processed_data['pixel_values'] = image
        processed_data['pixel_masks'] = pixel_mask
        if target is not None:
            target['masks'] = mask
            target['boxes'] = boxes
            processed_data['labels'] = target

        return processed_data

    def augment(self, img: torch.tensor, mask: torch.tensor, boxes: Union[torch.tensor, None]=None):
        if self.aug_backend == 'numpy':
            args = {
                'image': img.permute(1, 2, 0).numpy(),
                'mask': mask.permute(1, 2, 0).numpy()
            }
            if boxes is not None:
                args['bboxes'] = boxes.numpy()

            transformed = self.augmentator(**args)
            img = transformed['image'].transpose(2, 0, 1)
            mask = transformed['mask'].transpose(2, 0, 1)
            boxes = transformed.get('bboxes', boxes)

        else:
            # TODO: adapt for boxes as above
            transformed = self.augmentator(image=img, mask=mask)
            img, mask = transformed['image'], transformed['mask']
        return img, mask, boxes
    
    def resize_keeping_aspect_ratio(self, img, mask=None, boxes=None):
        orig_h, orig_w = img.shape[-2:]
        target_h , target_w = self.size
        ratio_h, ratio_w = target_h /orig_h, target_w / orig_w
        if ratio_h >= ratio_w:
            new_h, new_w = orig_h * ratio_w, target_w
        else:
            new_h, new_w = target_h, orig_w * ratio_h
        new_h, new_w = int(new_h), int(new_w)

        # Resize the image
        img_resized = F.interpolate(
            img.unsqueeze(0), 
            size=(new_h, new_w), 
            mode='bilinear', 
            align_corners=False
        ).squeeze(0)
        
        if mask is not None:
        # Resize the mask
            mask_resized = F.interpolate(
                mask.unsqueeze(0), 
                size=(new_h, new_w), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)
        else:
            mask_resized = None

        # Adjust bounding boxes according to resize
        if boxes is not None:
            scale_w = new_w / orig_w
            scale_h = new_h / orig_h
            boxes_resized = boxes.clone()
            boxes_resized[:, [0, 2]] = boxes[:, [0, 2]] * scale_w  # x1, x2
            boxes_resized[:, [1, 3]] = boxes[:, [1, 3]] * scale_h  # y1, y2
        else:
            boxes_resized = None

        return img_resized, mask_resized, boxes_resized

    def pad_to_required_size(self, image, mask):
        # BBoxes are not needed to be processed here, since we pad bottom and right
        new_h, new_w = self.size
        orig_h, orig_w = image.shape[-2:]
        pad_bottom = new_h - orig_h
        pad_right = new_w - orig_w
        padding = (0, pad_right, 0, pad_bottom)

        img_padded = F.pad(image, padding, mode='constant', value=0)


        # Create binary mask (1 for image, 0 for padding)
        pixel_mask = torch.ones((orig_h, orig_w), dtype=torch.float32)
        pixel_mask = F.pad(pixel_mask, padding, mode='constant', value=0)

        if mask is not None:
            mask_padded = F.pad(mask, padding, mode='constant', value=0)
        else:
            mask_padded = mask
        return img_padded, pixel_mask, mask_padded


    def _get_mean(self, x):
        if self.use_imagenet_norm:
            return self._imagenet_mean
        return torch.mean(x, dim=(1,2)).reshape(3,1,1)
    
    def _get_std(self, x):
        if self.use_imagenet_norm:
            return self._imagenet_std
        return torch.std(x, dim=(1,2)).reshape(3,1,1)
        