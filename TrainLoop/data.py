import os

from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
import torchvision.transforms as T
import numpy as np
import torch
import PIL
from PIL import Image

np_dtype = np.float16
torch_dtype = torch.float16
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class SemanticSegmentationCOCODataset(Dataset):
    """Image (semantic) segmentation dataset.
    Draws data from COCO-style dataset
    Images and annotations should be places in the same direcotry with name specified by "root_dir" parameter
    Annotation file (in COCO json format) name is specified by "ann_file_name" parameter
    """

    def __init__(self, root_dir, ann_file_name, preprocessor):
        self.root_dir = root_dir
        self.preprocessor = preprocessor
        self.coco_dataset = COCO(os.path.join(root_dir, ann_file_name))
        self.cat_ids = self.coco_dataset.getCatIds()
        self.idx_mapper = {raw_idx:coco_idx for raw_idx, coco_idx in enumerate(sorted(self.coco_dataset.imgs.keys()))}
        
    def __len__(self):
        return len(self.coco_dataset.imgs)

    def __getitem__(self, idx):
        # описание изображения в формате COCO
        coco_img = self.coco_dataset.imgs[self.idx_mapper[idx]]
        image = Image.open(os.path.join(self.root_dir, coco_img['file_name']))

        if (len(image.size) < 3) or (image.format != "JPEG"):
            image = image.convert('RGB')
        # достаем ID аннотаций этого изображения, затем сами аннотации
        coco_ann_ids = self.coco_dataset.getAnnIds(imgIds=coco_img['id'], catIds=self.cat_ids, iscrowd=None)
        coco_anns = self.coco_dataset.loadAnns(coco_ann_ids)
        
        segmentation_map = np.zeros((coco_img['height'],coco_img['width'])).astype(np_dtype)
        for ann in coco_anns:
            try:
                ann_mask = self.coco_dataset.annToMask(ann)
            except IndexError:
                continue
            segmentation_map = np.maximum(segmentation_map, (ann_mask*ann['category_id']).astype(np_dtype))
        return self.preprocessor(image, segmentation_map)
    
    

class ConvNextPreprocessorNumpy:
    def __init__(self, size, aug=None, use_imagenet_norm=True):
        self.aug = aug
        self.size = size
        # \_(^_^)_|
        if use_imagenet_norm:
            self.mean = np.array(IMAGENET_MEAN).reshape(3,1,1)
            self.std = np.array(IMAGENET_STD).reshape(3,1,1)
            self.get_mean = lambda x: self.mean
            self.get_std = lambda x: self.std
        else:
            self.get_mean = self._get_mean
            self.get_std = self._get_std
        
    def __call__(self, img: PIL.Image, mask: np.ndarray):

        img = self._preprocess_image(img)
        mask = self._preprocess_mask(mask)
        return img, mask
        
    def _preprocess_image(self, img):
        img = img.resize(self.size, resample=PIL.Image.Resampling.BILINEAR)
        img = np.array(img).transpose(2,0,1) # to CHW
        img = (img - self.get_mean(img)) / self.get_std(img)
        return img
    
    def _preprocess_mask(self, mask):
        mask = Image.fromarray(mask.astype(np.uint16))
        mask = np.array(mask.resize(self.size, resample=PIL.Image.NEAREST))
        return mask
        
    @staticmethod
    def _get_mean(x: np.ndarray):
        return np.mean(x, axis=(1,2)).reshape(3,1,1)
    
    @staticmethod
    def _get_std(x: np.ndarray):
        return np.std(x, axis=(1,2)).reshape(3,1,1)
    
    @staticmethod
    def _validate_img(img: PIL.Image):
        # Need testing. For now this handles png and single-channel images
        if img.format != "JPEG":
            img = img.convert('RGB')
        return img


class ConvNextPreprocessor:
    def __init__(self, size, aug=None, use_imagenet_norm=True):
        self.aug = aug
        self.size = size
        # \_(^_^)_|
        if use_imagenet_norm:
            self.mean = np.array(IMAGENET_MEAN).reshape(3,1,1)
            self.std = np.array(IMAGENET_STD).reshape(3,1,1)
            self.get_mean = lambda x: self.mean
            self.get_std = lambda x: self.std
        else:
            self.get_mean = self._get_mean
            self.get_std = self._get_std
        
    def __call__(self, img: PIL.Image, mask: np.ndarray):
        img = self._preprocess_image(img)
        mask = self._preprocess_mask(mask)
        return img, mask
        
    def _preprocess_image(self, img):
        img = T.functional.to_tensor(img).unsqueeze(0)
        img = T.functional.resize(img, size=self.size, interpolation=T.InterpolationMode.BILINEAR, antialias=True).squeeze()
        img = (img - self.get_mean(img)) / self.get_std(img)

        return img
    
    def _preprocess_mask(self, mask):
        mask = torch.tensor(mask).type(torch.int16)[None,None,:]
        mask = T.functional.resize(mask, size=self.size, interpolation=T.InterpolationMode.NEAREST, antialias=True).squeeze()
        return mask
        
    @staticmethod
    def _get_mean(x: np.ndarray):
        return torch.mean(x, dim=(1,2)).reshape(3,1,1)
    
    @staticmethod
    def _get_std(x: np.ndarray):
        return torch.std(x, dim=(1,2)).reshape(3,1,1)
