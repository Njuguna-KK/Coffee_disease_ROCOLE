"""Defines the neural network, losss function and metrics"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

pretrained_map = {
    "alexnet": models.alexnet(pretrained=True),
    "resnet": models.resnet18(pretrained=True),
    "resnext50":  models.resnext50_32x4d(pretrained=True)
}

class Regression_Adopted_NN(nn.Module):

    def get_num_outputs(self, params):
        if params.which_net == "alexnet":
            return pretrained_map[params.which_net].classifier[6].out_features
        else:
            return pretrained_map[params.which_net].fc.out_features

    def __init__(self, params):
        super(Regression_Adopted_NN, self).__init__()
        self.pretrained = pretrained_map[params.which_net]
        self.dropout_rate = params.dropout_rate if hasattr(params, 'dropout_rate') else 0.0
        self.my_new_layers = nn.Sequential(nn.Linear(self.get_num_outputs(params), 100, bias=True),
                                           nn.ReLU(),
                                           # nn.Dropout(p=self.dropout_rate),
                                           nn.Linear(100, 25, bias=True),
                                           nn.ReLU(),
                                           # nn.Dropout(p=self.dropout_rate),
                                           nn.Linear(25, 1, bias=True))


    def forward(self, s):
        s = self.pretrained(s)
        s = self.my_new_layers(s)
        return s


def loss_fn(outputs, labels):
    outputs = torch.flatten(outputs.type(torch.DoubleTensor))
    labels = labels.type(torch.DoubleTensor)
    criterion = nn.MSELoss()
    return criterion(outputs, labels)


def accuracy(outputs, labels):
    outputs = np.floor(outputs + 0.5).flatten()
    return np.sum(outputs == labels) / float(labels.size)


# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'accuracy': accuracy,
    # could add more metrics such as accuracy for each token type
}
