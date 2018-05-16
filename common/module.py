import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class Module(nn.Module):
    def __init__(self, config):
        super(Module, self).__init__()
        self.config = config

    def init_weight(self):
        raise NotImplementedError()

    def fix_params(self):
        raise NotImplementedError()

    def get_pred_names(self, is_train):
        raise NotImplementedError()

    def get_label_names(self):
        raise NotImplementedError()

    def check_parameters_shape(self, is_train):
        raise NotImplementedError()

    def forward(self, *inputs, **kwargs):
        if self.training:
            return self.train_forward(*inputs, **kwargs)
        else:
            return self.inference_forward(*inputs, **kwargs)

    def train_forward(self, data, label, **kwargs):
        outputs = None
        loss = 0
        return [outputs], loss

    def inference_forward(self, data, **kwargs):
        outputs = None
        return [outputs]

