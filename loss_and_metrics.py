from sklearn import metrics as sklearn_metrics
import numpy as np
import torch.nn as nn


def loss_fn(outputs, labels):
    criterion = nn.NLLLoss()
    return criterion(outputs, labels)


def macro_f1(outputs, labels):
    outputs = np.argmax(outputs, axis=1)
    macro_f1 = sklearn_metrics.f1_score(labels, outputs, average='macro')
    return macro_f1


def macro_precision(outputs, labels):
    outputs = np.argmax(outputs, axis=1)
    precision_score = sklearn_metrics.precision_score(labels, outputs, average='macro')
    return precision_score


def macro_recall(outputs, labels):
    outputs = np.argmax(outputs, axis=1)
    recall_score = sklearn_metrics.recall_score(labels, outputs, average='macro')
    return recall_score


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
    'macro f1': macro_f1,
    'macro precision': macro_precision,
    'macro recall': macro_recall
    # could add more metrics such as accuracy for each token type
}
