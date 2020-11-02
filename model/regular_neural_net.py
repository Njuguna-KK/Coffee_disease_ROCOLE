"""Defines the neural network, losss function and metrics"""

import numpy as np
import torch.nn as nn
import torch

class Net(nn.Module):
    def __init__(self, params):
        """
        We define a fully connected network that predicts the category of disease of coffee leaves:
            LINEAR -> RELU  -> LINEAR -> RELU -> LINEAR -> SOFTMAX

        Args:
            params: (Params) contains num_channels
        """
        super(Net, self).__init__()
        self.num_classes = params.num_classes
        self.img_dimension = params.img_dimension
        self.first_hidden_size = params.first_hidden_size
        self.second_hidden_size = params.second_hidden_size
        self.third_hidden_size = params.third_hidden_size
        self.depth = params.depth
        self.batch_size = params.batch_size

        self.linear1 = nn.Linear(self.img_dimension * self.img_dimension * self.depth, self.first_hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(self.first_hidden_size, self.second_hidden_size)
        self.relu = nn.ReLU()
        self.linear3 = nn.Linear(self.second_hidden_size, self.third_hidden_size)
        self.relu = nn.ReLU()
        self.linear4 = nn.Linear(self.third_hidden_size, self.num_classes)

    def forward(self, x):
        num_examples = x.shape[0]
        out = x.view(num_examples, -1)
        out = self.linear1(out)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        out = self.relu(out)
        out = self.linear4(out)

        return out

def loss_fn(outputs, labels):
    """
    This criterion combines LogSoftmax and NLLLoss in one single class.
    """
    criterion = nn.CrossEntropyLoss()
    return criterion(outputs, labels)

def accuracy(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels for all images.

    Args:
        outputs: (np.ndarray) dimension batch_size x 6 - log softmax output of the model
        labels: (np.ndarray) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5]

    Returns: (float) accuracy in [0,1]
    """
    outputs = np.argmax(outputs, axis=1)
    return np.sum(outputs==labels)/float(labels.size)

# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'accuracy': accuracy,
    # could add more metrics such as accuracy for each token type
}