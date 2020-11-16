"""Defines the neural network, losss function and metrics"""

import torch.nn as nn
import utils


class Net(nn.Module):

    def __init__(self, args, params):
        super(Net, self).__init__()
        self.pretrained = utils.get_model(args)
        self.dropout_rate = params.dropout_rate if hasattr(params, 'dropout_rate') else 0.0
        self.my_new_layers = nn.Sequential(nn.Linear(utils.get_num_outputs(args), 128, bias=True),
                                           nn.ReLU(),
                                           nn.Dropout(p=self.dropout_rate),
                                           nn.Linear(128, params.num_classes, bias=True),
                                           nn.LogSoftmax(dim=1))


    def forward(self, s):
        s = self.pretrained(s)
        s = self.my_new_layers(s)
        return s
