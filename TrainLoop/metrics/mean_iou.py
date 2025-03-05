from typing import Dict, Optional

import numpy as np

def total_intersect_and_union(
    results,
    gt_seg_maps,
    num_labels,
    ignore_index: bool,
    label_map: Optional[Dict[int, int]] = None,
    reduce_labels: bool = False,
):
    """Calculate Total Intersection and Union, by calculating `intersect_and_union` for each (predicted, ground truth) pair.

    Args:
        results (`ndarray`):
            List of prediction segmentation maps, each of shape (height, width).
        gt_seg_maps (`ndarray`):
            List of ground truth segmentation maps, each of shape (height, width).
        num_labels (`int`):
            Number of categories.
        ignore_index (`int`):
            Index that will be ignored during evaluation.
        label_map (`dict`, *optional*):
            Mapping old labels to new labels. The parameter will work only when label is str.
        reduce_labels (`bool`, optional, defaults to `False`):
            Whether or not to reduce all label values of segmentation maps by 1. Usually used for datasets where 0 is used for background,
            and background itself is not included in all classes of a dataset (e.g. ADE20k). The background label will be replaced by 255.

     Returns:
         total_area_intersect (`ndarray`):
            The intersection of prediction and ground truth histogram on all classes.
         total_area_union (`ndarray`):
            The union of prediction and ground truth histogram on all classes.
         total_area_pred_label (`ndarray`):
            The prediction histogram on all classes.
         total_area_label (`ndarray`):
            The ground truth histogram on all classes.
    """
    total_area_intersect = np.zeros((num_labels,), dtype=np.float64)
    total_area_union = np.zeros((num_labels,), dtype=np.float64)
    total_area_pred_label = np.zeros((num_labels,), dtype=np.float64)
    total_area_label = np.zeros((num_labels,), dtype=np.float64)
    for result, gt_seg_map in zip(results, gt_seg_maps):
        area_intersect, area_union, area_pred_label, area_label = intersect_and_union(
            result, gt_seg_map, num_labels, ignore_index, label_map, reduce_labels
        )
        total_area_intersect += area_intersect
        total_area_union += area_union
        total_area_pred_label += area_pred_label
        total_area_label += area_label
    return total_area_intersect, total_area_union, total_area_pred_label, total_area_label

def intersect_and_union(
    pred_label,
    label,
    num_labels,
    ignore_index: bool,
    label_map: Optional[Dict[int, int]] = None,
    reduce_labels: bool = False,
):
    """Calculate intersection and Union.

    Args:
        pred_label (`ndarray`):
            Prediction segmentation map of shape (height, width).
        label (`ndarray`):
            Ground truth segmentation map of shape (height, width).
        num_labels (`int`):
            Number of categories.
        ignore_index (`int`):
            Index that will be ignored during evaluation.
        label_map (`dict`, *optional*):
            Mapping old labels to new labels. The parameter will work only when label is str.
        reduce_labels (`bool`, optional, defaults to `False`):
            Whether or not to reduce all label values of segmentation maps by 1. Usually used for datasets where 0 is used for background,
            and background itself is not included in all classes of a dataset (e.g. ADE20k). The background label will be replaced by 255.

     Returns:
         area_intersect (`ndarray`):
            The intersection of prediction and ground truth histogram on all classes.
         area_union (`ndarray`):
            The union of prediction and ground truth histogram on all classes.
         area_pred_label (`ndarray`):
            The prediction histogram on all classes.
         area_label (`ndarray`):
            The ground truth histogram on all classes.
    """
    if label_map is not None:
        for old_id, new_id in label_map.items():
            label[label == old_id] = new_id

    # turn into Numpy arrays
    pred_label = np.array(pred_label)
    label = np.array(label)

    if reduce_labels:
        label[label == 0] = 255
        label = label - 1
        label[label == 254] = 255

    mask = label != ignore_index
    mask = np.not_equal(label, ignore_index)
    pred_label = pred_label[mask]
    label = np.array(label)[mask]

    intersect = pred_label[pred_label == label]

    area_intersect = np.histogram(intersect, bins=num_labels, range=(0, num_labels - 1))[0]
    area_pred_label = np.histogram(pred_label, bins=num_labels, range=(0, num_labels - 1))[0]
    area_label = np.histogram(label, bins=num_labels, range=(0, num_labels - 1))[0]

    area_union = area_pred_label + area_label - area_intersect

    return area_intersect, area_union, area_pred_label, area_label


class MeanIOU:
    def __init__(self,
                 num_labels,
                 mode='accum',
                 ignore_index: Optional[int] = None,
                 label_map: Optional[Dict[int, int]] = None,
                 reduce_labels: bool = False,
                 nan_to_num: Optional[int] = None,
):
        '''
        Args:
            num_labels (`int`):
                Number of categories.
            mode: str = ['at_once', 'accum']. How input will be provided:
                'at_once' - all validation targets and preds at once
                'accum' - iterative regime. Targets ans preds will be provided in batch
            ignore_index (`int`):
                Index that will be ignored during evaluation.

            label_map (`dict`, *optional*):
                Mapping old labels to new labels. The parameter will work only when label is str.
            reduce_labels (`bool`, optional, defaults to `False`):
                Whether or not to reduce all label values of segmentation maps by 1. Usually used for datasets where 0 is used for background,
                and background itself is not included in all classes of a dataset (e.g. ADE20k). The background label will be replaced by 255.
            nan_to_num (`int`, *optional*):
                If specified, NaN values will be replaced by the number defined by the user.
        '''
        assert mode in ['at_once', 'accum'], 'parameter "mode" should be either "at_once" or "accum"'
        self.mode = mode
        self.num_labels = num_labels
        self.ignore_index = ignore_index
        self.label_map = label_map
        self.reduce_labels = reduce_labels
        self.nan_to_num = nan_to_num
        
        self._refresh_accumulations()
        
    def compute(
        self,
        results,
        gt_seg_maps,
    ):
        """Calculate Mean Intersection and Union (mIoU).

        Args:
            results (`ndarray`):
                List of prediction segmentation maps, each of shape (height, width).
            gt_seg_maps (`ndarray`):
                List of ground truth segmentation maps, each of shape (height, width).
            

        Returns:
            `Dict[str, float | ndarray]` comprising various elements:
            - mean_iou (`float`):
                Mean Intersection-over-Union (IoU averaged over all categories).
            - mean_accuracy (`float`):
                Mean accuracy (averaged over all categories).
            - overall_accuracy (`float`):
                Overall accuracy on all images.
            - per_category_accuracy (`ndarray` of shape `(num_labels,)`):
                Per category accuracy.
            - per_category_iou (`ndarray` of shape `(num_labels,)`):
                Per category IoU.
        """
        total_area_intersect, total_area_union, total_area_pred_label, total_area_label = total_intersect_and_union(
            results, gt_seg_maps, self.num_labels, self.ignore_index, self.label_map, self.reduce_labels
        )
        self.total_area_intersect += total_area_intersect 
        self.total_area_union += total_area_union 
        self.total_area_pred_label += total_area_pred_label 
        self.total_area_label += total_area_label
        if self.mode == 'at_once':
            return self.get_results()

    
    def get_results(self):
        # compute metrics
        metrics = dict()

        all_acc = self.total_area_intersect.sum() / self.total_area_label.sum()
        iou = self.total_area_intersect / self.total_area_union
        acc = self.total_area_intersect / self.total_area_label

        metrics["mean_iou"] = np.nanmean(iou)
        metrics["mean_accuracy"] = np.nanmean(acc)
        metrics["overall_accuracy"] = all_acc
        metrics["per_category_iou"] = iou
        metrics["per_category_accuracy"] = acc

        if self.nan_to_num is not None:
            metrics = dict(
                {metric: np.nan_to_num(metric_value, nan=self.nan_to_num) for metric, metric_value in metrics.items()}
            )
        self._refresh_accumulations()
        return metrics
    
    def _refresh_accumulations(self):
        self.total_area_intersect = 0
        self.total_area_union = 0
        self.total_area_pred_label = 0
        self.total_area_label = 0