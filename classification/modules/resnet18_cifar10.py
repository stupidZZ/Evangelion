import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import common.blocks.backbone.resnet as resnet

from common.module import Module

class resnet18_cifar10(Module):
    def __init__(self, config):
        super(resnet18_cifar10, self).__init__(config)
        self.resnet18 = resnet.ResNet18(10)
        self.criterion = nn.CrossEntropyLoss()

    def get_pred_names(self, is_train):
        return ["cls_prob"]

    def get_label_names(self, is_train):
        return ["labels"]

    def train_forward(self, data, label, **kwargs):
        output = self.resnet18(data)
        loss = self.criterion(output, label)
        return [output], loss.reshape((-1,))

    def inference_forward(self, data, **kwargs):
        output = self.resnet18(data)
        return [output]

