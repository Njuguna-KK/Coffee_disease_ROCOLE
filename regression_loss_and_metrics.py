from sklearn import metrics as sklearn_metrics
import numpy as np
import torch.nn as nn
import torch


def regression_loss_fn(outputs, labels):
    outputs = torch.flatten(outputs.type(torch.DoubleTensor))
    labels = labels.type(torch.DoubleTensor)
    criterion = nn.MSELoss()
    return criterion(outputs, labels)


def regression_accuracy(outputs, labels):
    outputs = np.floor(outputs + 0.5).flatten()
    return np.sum(outputs == labels) / float(labels.size)


def macro_f1(outputs, labels):
    outputs = np.floor(outputs + 0.5).flatten()
    macro_f1 = sklearn_metrics.f1_score(labels, outputs, average='macro')
    return macro_f1


def macro_precision(outputs, labels):
    outputs = np.floor(outputs + 0.5).flatten()
    precision_score = sklearn_metrics.precision_score(labels, outputs, average='macro')
    return precision_score


def macro_recall(outputs, labels):
    outputs = np.floor(outputs + 0.5).flatten()
    recall_score = sklearn_metrics.recall_score(labels, outputs, average='macro')
    return recall_score


regression_metrics = {
    'accuracy': regression_accuracy,
    'macro f1': macro_f1,
    'macro precision': macro_precision,
    'macro recall': macro_recall
}