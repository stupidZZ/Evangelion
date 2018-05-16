import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from common.module import Module

class plain_conv(Module):
    def __init__(self, config):
        super(plain_conv, self).__init__(config)
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.criterion = nn.CrossEntropyLoss()

    def get_pred_names(self, is_train):
        return ["cls_prob"]

    def get_label_names(self, is_train):
        return ["labels"]

    def train_forward(self, data, label, **kwargs):
        x = self.pool(F.relu(self.conv1(data)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = self.fc3(x)
        loss = self.criterion(output, label)
        return [output], loss.reshape((-1,))

    def inference_forward(self, data, **kwargs):
        x = self.pool(F.relu(self.conv1(data)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = self.fc3(x)
        return [output]

