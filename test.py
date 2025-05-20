import torch
import requests
from PIL import Image
import numpy as np
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation, Mask2FormerConfig

# load Mask2Former fine-tuned on COCO instance segmentation
processor = AutoImageProcessor.from_pretrained("./models/mask2former")
model = Mask2FormerForUniversalSegmentation.from_pretrained('models/mask2former', )

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
image = Image.open('a.jpg')
x = processor(image, return_tensors='pt')
model.eval()
with torch.no_grad():
    res = model(**x)


pred_instance_map = processor.post_process_instance_segmentation(
    res, target_sizes=[(image.height, image.width)]
)[0]
print('---', pred_instance_map)

import matplotlib.pyplot as plt
mask = pred_instance_map['segmentation'].cpu().numpy()
print(np.unique(mask))
plt.imsave('aboba.jpg', mask)