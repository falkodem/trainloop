from collections import OrderedDict, defaultdict

import numpy as np
import evaluate
import matplotlib.pyplot as plt
import cv2
import joblib

def threshold_img(img, thr=0.9):
    if img.max() <= 1.:
        img = (img*255).astype(np.uint8).copy()
    _, thresh = cv2.threshold(img, int(thr*255), 255, 0)
    return thresh

def split_masks(img):
    '''
    Разделяет объекты (замкнутые контуры) бинарного изображения в виде масок по каналам. Например: 3 объекта на изображении (H,W).
    Функция вернет массив (3, H, W), где в каждом канале будет маска только 1 объекта
    
    Args:
        img (np.ndarray): binarized image of shape (h,w), np.uint8        
    Returns:
        masks (np.ndarray): array with shape (N,H,W) where N is number of found objects/contours. 
        Array contains masks of objects/contours 
    '''
    method = 'connected_comps'
    if method == 'connected_comps':
        n_comps, comps = cv2.connectedComponentsWithAlgorithm(img, 8, ltype=cv2.CV_16U, ccltype=cv2.CCL_BOLELLI)
#         n_comps = max(1, n_comps - 1)
        n_comps -= 1
        objs_mask = np.zeros((n_comps, *img.shape), dtype=np.uint8)
        for obj_idx in range(n_comps):
            objs_mask[obj_idx] = (comps == obj_idx+1).astype(np.uint8) 
    else:
        contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        objs_mask = np.zeros((len(contours), *img.shape))
        for i, cnt in enumerate(contours):
            cv2.fillPoly(objs_mask[i], pts =[cnt], color=(1))
    return objs_mask

def maskwise_iou(mask1, mask2, eps=1e-10):
    '''
    Считает IUO для масок двух изображений. Каждый массив должен быть размера (BS, C, H, W), где
        BS - размер батчка
        C - каналы, под каждый замкнутый контур маски - свой канал
        H, W - высота и ширина
    IUO считается для масок "каждый-с-каждым" между элементами двух массивов в одинаковым батчевым индексом
    
    Params:
        mask1 (np.ndarray): first mask array
        mask1 (np.ndarray): second mask array
        
    Returns:
        matrix of pairwise iou's for masks
    '''
    # mask1: (BS, C1, H, W) -> (BS, C1, 1, H, W)
    # mask2: (BS, C2, H, W) -> (BS, 1, C2, H, W)
    mask1, mask2 = np.expand_dims(mask1, 2).astype(bool), np.expand_dims(mask2, 1).astype(bool)
    # Магия Numpy Broadcasting ..
    # сотворяет (BS, C1, C2, H, W). Затем сумма по осям H, W -> (BS, C1, C2)
    inter = np.sum((mask1 & mask2), axis=(3,4))
    union = np.sum((mask1 | mask2), axis=(3,4))
    return inter / (union + eps)

def get_binary_metrics(iou_mat, iou_thr=0.5):
    '''
    Функция, которая по матрице попарных IOU выносит решение о том, произошла ли правильная детекция. Затем считает метрики
    Отличие этого метода в том, что если более одного предсказанного bounding box'а пересекаются с одним 
    ground truth bounding box'ом, так, что сумма их IOU превышает порог, то это считается за попадание - true positive.
    Аналогично и обратное: если один предсказанные bounding box пересекается с более чем одним ground truth bounding box'ом
    так, что сумма их IOU превышает порог, то это считается за попадание - true positive.

    Params:
        iou_mat (np.ndarray): pairwise IOU matrix, dim0 is batch_size, dim1 is ground turth, dim2 is prediction
        iou_thr (float): threshold for sum iou to consider match is true positive
        
    Returns:
        f1_score, precision, recall (tuple[float]): metrics
        
    '''
    recall = np.mean(np.sum(iou_mat, axis=1) > iou_thr)
    precision = np.mean(np.sum(iou_mat, axis=0) > iou_thr)
    if np.isnan(precision):
        precision = 0
    if np.isnan(recall):
        recall = 0
    return precision, recall


def get_tps(iou_mat, iou_thr=0.5, eps=1e-10):
    '''
    Функция, которая по матрице попарных IOU выносит решение о том, произошла ли правильная детекция. Затем считает кол-во true positive
    Отличие этого метода в том, что если более одного предсказанного bounding box'а пересекаются с одним 
    ground truth bounding box'ом, так, что сумма их IOU превышает порог, то это считается за попадание - true positive.
    Аналогично и обратное: если один предсказанный bounding box пересекается с более чем одним ground truth bounding box'ом
    так, что сумма их IOU превышает порог, то это считается за попадание - true positive.
    Также из-за особенности всего алгоритма, считаем true positive отдельно для recall и отдельно для precision

    Params:
        iou_mat (np.ndarray): pairwise IOU matrix, dim0 is batch_size, dim1 is ground turth, dim2 is prediction
        iou_thr (float): threshold for sum iou to consider match is true positive
        
    Returns:
        tp4pricision, tp4recall (tuple(int)): true positives for precision and recall calculation separetely
        
    '''
    tp4recall = np.sum(np.sum(iou_mat, axis=2) > iou_thr, axis=1)
    tp4precision = np.sum(np.sum(iou_mat, axis=1) > iou_thr, axis=1)
    return tp4precision, tp4recall



def compute_ap(recall, precision):
    """Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
    # Returns
        Average precision, precision curve, recall curve.
    """
    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    # Integrate area under curve
    method = "interp"  # methods: 'continuous', 'interp'
    if method == "interp":
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec

def split_masks_in_batch(label_masks):
    '''
    Функция получается батч из двухмерных матриц, содержащих бинарную маску (в контексте всего алгоритма - это маска какого-либо класса)
    Для каждой матрицы находим замкнутые контуры - это "объекты". Каждый "объект" кладем в отдельную двухмерную матрицу - канал
    На выходе массив размера (batch_size, channels, H, W),
        где channels - это максимальное количество каналов, другими словами: максимальное найденное количество замкнутых контуров среди матриц
    
    Например: имеем батч из 5 матриц с масками, размерность массива (5, H, W).
    Среди матриц максимальное количество найденных замкнутых контуров - 7.
    Выходной массив имеет размер (5, 7, H, W)
    Для элемента батча, у которого кол-во найденных замкнутых контуров N меньше максимального по батчу N_MAX, заполняются первые N каналов в выходном массиве (остальные каналы нулевые)
    Поэтому также возвращаем индекс N последнего ненулевого канала для каждого элемента батча
    
    Params: 
        label_masks: batch of binary masks (batch_size, H, W) 
        
    Returns:
        label_mask_split_array (np.ndarray): Batch of splitted binary masks of shape (batch_size, channels, H, W), each found component occupies its own channel
        num_of_comps (list): list of shape (batch_size) indicating index of the last channel that is filled with component (channel index of the last non-zero matrix (H,W))
            for each element in batch. In other words - number of components for each value in batch
    '''
    label_mask_split = []
    for mask in label_masks:
        label_mask_split.append(split_masks(mask))
    
    # Количество компонент
    num_of_comps = [masks.shape[0] for masks in label_mask_split]
    max_comps_in_batch = max(num_of_comps)
    # Создаем массив (размер_батча, макс_компонент_в_батче, H, W)
    label_mask_split_array = np.zeros([label_masks.shape[0], max_comps_in_batch, *label_masks.shape[-2:]])
    for i_batch, masks in enumerate(label_mask_split):
        # Для каждого элемента батча, заполняем столько масок, сколько получилось после функции split_masks. Остальные будет нулевые
        label_mask_split_array[i_batch, :masks.shape[0], :, :] = masks
    return label_mask_split_array, num_of_comps

# Это импортить для tp4prec, tp4rec, preds_num_mask, gts_num_masks. По большому счету эта функция нужна для расчета map, если нужны просто precision и recall, используй get_binary_metrics().
def get_metrics_base(gts, preds, thr_iou=0.5, background_idx=None, eps=1e-10, return_accumulatable_metrics=True):
    '''
    TO BE DONE...
    Params:
    ...

    return_accumulatable_metrics: (bool) - Если True, то эта функция должна использоваться для расчитывания метрик для батчей, т.е. она возвращает результаты, которые можно корректро накапливать для множества батчей. И самое главное - использовать для расчета MAP. Если False - то, функция работает в режиме "расчет сразу на всем тестовом датасете"
    '''
    metrics = {}
    gt_label_names = np.unique(gts)
    pred_label_names = np.unique(preds)
    label_names = set(gt_label_names) | set(pred_label_names)
    for label_name in label_names:
        # Фон считаем как один "особый" объект
        if label_name == background_idx:
            preds_lab_mask = (preds == label_name)
            gts_lab_mask = (gts == label_name)
            tp = np.sum(preds_lab_mask & gts_lab_mask)
            fn = np.sum(~preds_lab_mask & gts_lab_mask)
            fp = np.sum(preds_lab_mask & ~gts_lab_mask)
            if return_accumulatable_metrics:
                metrics[label_name] = np.array([tp, tp, tp+fp, tp+fn])
            else:
                metrics[label_name] = np.array([tp/(tp+fp), tp/(tp+fn)])
        # Остальные классы обрабытваем сложнее
        else:
            gts_label_masks = (gts==label_name).astype(np.uint8)
            preds_label_masks = (preds==label_name).astype(np.uint8)
            # Обработка ground_truth
            gts_lab_masks, gts_num_masks = split_masks_in_batch(gts_label_masks)
            # Обработка preds
            preds_lab_masks, preds_num_masks = split_masks_in_batch(preds_label_masks)    
            iou_matrix = maskwise_iou(gts_lab_masks, preds_lab_masks)
            if return_accumulatable_metrics:
                tp4prec, tp4rec = get_tps(iou_matrix, thr_iou)
                metrics[label_name] = np.array(list(map(sum, [tp4prec, tp4rec, preds_num_masks, gts_num_masks])))
            else:
                prec, rec = get_binary_metrics(iou_matrix, thr_iou)
                metrics[label_name] = np.array([prec, rec])
            
    return metrics

class MetricAccumulator:
    def __init__(self, iou_steps: np.ndarray, thr_steps: np.ndarray):
        self.thr_steps = thr_steps
        self.iou_steps = iou_steps
        self.met_mat = defaultdict(lambda: np.zeros((len(self.iou_steps), len(self.thr_steps), 4)))
        self.curr_iou_step = None
        self.curr_thr_step = None

    def compute(self, blacklist: list = []):
        res = {}
        mean_ap = {}
        for iou_step_idx, iou_step in enumerate(self.iou_steps):
            ap = {}
            for label, stats in self.met_mat.items():
                if label not in blacklist:
                    precision = stats[iou_step_idx, :, 0] / (stats[iou_step_idx, :, 2] + 1e-10)
                    recall = stats[iou_step_idx, :, 1] / (stats[iou_step_idx, :, 3] + 1e-10)
                    ap[label] = compute_ap(recall[::-1], precision[::-1])[0]
            mean_ap[iou_step] = np.mean([p for p in ap.values()])
        return mean_ap
    
    def append(self, batch_stats, iou_step, thr_step):
        idx_iou_step = np.where(self.iou_steps == iou_step)[0].item()
        idx_thr_step = np.where(self.thr_steps == thr_step)[0].item()
        
        if (iou_step != self.curr_iou_step) or (thr_step != self.curr_thr_step):
            for label, label_batch_stats in batch_stats.items():
                self.met_mat[label][idx_iou_step, idx_thr_step] = label_batch_stats
        else:
            for label, label_batch_stats in batch_stats.items():
                self.met_mat[label][idx_iou_step, idx_thr_step] += label_batch_stats            
                
        self.curr_iou_step = iou_step
        self.curr_thr_step = thr_step

# Это импортить для mean average precision
def get_agg_metrics(data_generator, background_idx=None, eps=1e-10):
    metrics = {'label': [],
               f'mean_precision': [],
               f'mean_recall': [],
               f'mean_f1': [],
               'ap50': [],
               'ap50-95': []}
    
    thr_iou_steps = np.round(np.arange(0.5, 0.95, 0.05), 2)
    thr_conf_steps = np.arange(0.9, 1., 0.01)
    metric_accum = MetricAccumulator(thr_iou_steps, thr_conf_steps)
    for thr_iou_step in thr_iou_steps:
        for thr_conf_idx, thr_conf_step in enumerate(thr_conf_steps):
            for i_batch, (preds, gts) in enumerate(data_generator):
                # Откидываем все предикты, ниже порога отсечки, добавляем еще один канал для фона, считаем наибольший скор
                # preds_label = np.argmax(np.concatenate([np.zeros((preds.shape[0], 1, *preds.shape[-2:]), dtype=np.float32) + thr_conf_step,
                #                np.where(preds > thr_conf_step, preds, 0)], axis=1), axis=1)
                preds_label = np.argmax(np.where(preds > thr_conf_step, preds, 0), axis=1)
                metrics_batch = get_metrics_base(gts, preds_label, thr_iou_step, background_idx=background_idx, return_accumulatable_metrics=True)
                metric_accum.append(metrics_batch, thr_iou_step, thr_conf_step)                
            data_generator.reset()
    res = metric_accum.compute(blacklist=[])
    return res