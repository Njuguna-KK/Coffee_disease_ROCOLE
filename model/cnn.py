"""Defines the neural network, losss function and metrics"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    """
    This is the standard way to define your own network in PyTorch. You typically choose the components
    (e.g. LSTMs, linear layers etc.) of your network in the __init__ function. You then apply these layers
    on the input step-by-step in the forward function. You can use torch.nn.functional to apply functions

    such as F.relu, F.sigmoid, F.softmax, F.max_pool2d. Be careful to ensure your dimensions are correct after each
    step. You are encouraged to have a look at the network in pytorch/nlp/model/net.py to get a better sense of how
    you can go about defining your own network.

    The documentation for all the various components available o you is here: http://pytorch.org/docs/master/nn.html
    """

    def __init__(self, params):
        """
        We define an convolutional network that predicts the sign from an image. The components
        required are:

        - an embedding layer: this layer maps each index in range(params.vocab_size) to a params.embedding_dim vector
        - lstm: applying the LSTM on the sequential input returns an output for each token in the sentence
        - fc: a fully connected layer that converts the LSTM output for each token to a distribution over NER tags

        Args:
            params: (Params) contains num_channels
        """
        super(Net, self).__init__()
        self.num_channels = params.num_channels
        self.dropout_rate = params.dropout_rate

        # each of the convolution layers below have the arguments (input_channels, output_channels, filter_size,
        # stride, padding). We also include batch normalisation layers that help stabilise training.
        # For more details on how to use these layers, check out the documentation.
        self.conv0 = nn.Conv2d(3, 24, 11, stride=4, padding=2)
        self.bn0 = nn.BatchNorm2d(24)
        self.pool0 = nn.MaxPool2d(2, stride=2)

        self.conv1 = nn.Conv2d(24, 42, 5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(42)
        self.pool1 = nn.MaxPool2d(2, stride=2)

        self.conv2 = nn.Conv2d(42, 74, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(74)
        self.pool2 = nn.MaxPool2d(2, stride=2)

        self.conv3 = nn.Conv2d(74, 148, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(148)
        self.pool3 = nn.MaxPool2d(2, stride=2)

        # 2 fully connected layers to transform the output of the convolution layers to the final output
        self.fc1 = nn.Linear(1332, 600)
        self.fc2 = nn.Linear(600, 200)
        self.fc3 = nn.Linear(200, 6)

    def forward(self, s):
        """
        This function defines how we use the components of our network to operate on an input batch.

        Args:
            s: (Variable) contains a batch of images, of dimension batch_size x 3 x 64 x 64 .

        Returns:
            out: (Variable) dimension batch_size x 6 with the log probabilities for the labels of each image.

        Note: the dimensions after each step are provided
        """
        s = self.conv0(s)
        s = self.bn0(s)
        s = self.pool0(s)
        s = F.relu(s)

        s = self.conv1(s)
        s = self.bn1(s)
        s = self.pool1(s)
        s = F.relu(s)

        s = self.conv2(s)
        s = self.bn2(s)
        s = self.pool2(s)
        s = F.relu(s)
        s = self.conv3(s)
        s = self.bn3(s)
        s = self.pool3(s)
        s = F.relu(s)

        # flatten the output for each image
        s = s.view(-1, 1332)

        # print(s.shape)
        s = F.relu(self.fc1(s))
        s = F.dropout(F.relu(self.fc2(s)),
                      p=self.dropout_rate, training=self.training)
        s = self.fc3(s)

        return s


def loss_fn(outputs, labels):
    """
    Compute the cross entropy loss given outputs and labels.

    Args:
        outputs: (Variable) dimension batch_size x 6 - output of the model
        labels: (Variable) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5]

    Returns:
        loss (Variable): cross entropy loss for all images in the batch

    Note: you may use a standard loss function from http://pytorch.org/docs/master/nn.html#loss-functions. This example
          demonstrates how you can easily define a custom loss function.
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
    return np.sum(outputs == labels) / float(labels.size)


# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'accuracy': accuracy,
    # could add more metrics such as accuracy for each token type
}
