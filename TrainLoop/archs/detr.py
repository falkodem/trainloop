import torch.nn as nn
from transformers import DetrForSegmentation

class DetrSeg(nn.Module):
    def __init__(self, core_model_path):
        super(DetrSeg, self).__init__()
        self.core_model = DetrForSegmentation.from_pretrained(core_model_path)

    def forward(self, train_data):
        # train_data = {'pixel_values': X['pixel_values'],
        #               'pixel_mask': X['pixel_mask']}
        # if y is not None:
        #     train_data['labels'] = y

        return self.core_model(**train_data)