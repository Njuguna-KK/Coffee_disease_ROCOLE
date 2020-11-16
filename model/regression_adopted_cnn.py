"""Defines the neural network, losss function and metrics"""

import torch.nn as nn
import utils


class Regression_Adopted_NN(nn.Module):

    def __init__(self, params):
        super(Regression_Adopted_NN, self).__init__()
        self.pretrained = utils.get_model(params)
        self.dropout_rate = params.dropout_rate if hasattr(params, 'dropout_rate') else 0.0
        self.my_new_layers = nn.Sequential(nn.Linear(utils.get_num_outputs(params), 128, bias=True),
                                           nn.ReLU(),
                                           nn.Dropout(p=self.dropout_rate),
                                           nn.Linear(128, 36, bias=True),
                                           nn.ReLU(),
                                           nn.Dropout(p=self.dropout_rate),
                                           nn.Linear(36, 1, bias=True))


    def forward(self, s):
        s = self.pretrained(s)
        s = self.my_new_layers(s)
        return s