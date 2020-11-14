"""Evaluates the model"""

import argparse
import logging
import os

import numpy as np
import torch
from torch.autograd import Variable
import utils
import model.cnn as net
import model.data_loader as data_loader
import torchvision.models as models
import model.regression_adopted_cnn as regression_adopted_cnn
import model.regular_neural_net as regular_nn

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir',
                    help="Directory containing the dataset")
parser.add_argument('--model_dir',
                    help="Directory containing params.json")
parser.add_argument('--restore_file', default='best', help="name of the file in --model_dir \
                     containing weights to load")
parser.add_argument('--net',
                    help="Which neural net to use")

pretrained_map = {
    "alexnet": models.alexnet(pretrained=True),
    "resnet": models.resnet18(pretrained=True),
    "resnext50":  models.resnext50_32x4d(pretrained=True),
}


def get_model_loss_metrics(args, params):
    if args.net == 'regression':
        return regression_adopted_cnn.Regression_Adopted_NN(params), regression_adopted_cnn.loss_fn, regression_adopted_cnn.metrics
    if args.net == 'custom':
        return net.Net(params), net.loss_fn, net.metrics
    if args.net == 'regular':
        return regular_nn.Net(params), regular_nn.loss_fn, regular_nn.metrics
    else:
        return pretrained_map[args.net], net.loss_fn, net.metrics


def compute_and_save_f1(saved_outputs, saved_labels, file):
    conf_matrix, report = utils.f1_metrics(saved_outputs, saved_labels)

    text_file = open(file, "wt")
    text_file.write('Confusion matrix: \n {}\n\n Classification Report: \n {}'.format(conf_matrix, report))
    text_file.close()

def process_output(args, output_batch):
    if args.net != 'regression':
        res = np.argmax(output_batch, axis=1)
        return res
    return np.floor(output_batch + 0.5).flatten()




def evaluate(model, loss_fn, dataloader, metrics, params, which_set, file, args):
    """Evaluate the model on `num_steps` batches.

    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to evaluation mode
    model.eval()

    # summary for current eval loop
    summ = []

    saved_outputs = []
    saved_labels = []
    # compute metrics over the dataset
    for data_batch, labels_batch in dataloader:
        # move to GPU if available
        if params.cuda:
            data_batch, labels_batch = data_batch.cuda(
                non_blocking=True), labels_batch.cuda(non_blocking=True)
        # fetch the next evaluation batch
        data_batch, labels_batch = Variable(data_batch), Variable(labels_batch)

        # compute model output
        output_batch = model(data_batch)
        loss = loss_fn(output_batch, labels_batch)

        # extract data from torch Variable, move to cpu, convert to numpy arrays
        output_batch = output_batch.data.cpu().numpy()
        labels_batch = labels_batch.data.cpu().numpy()

        processed_output = process_output(args, output_batch)
        saved_outputs.extend(processed_output)
        saved_labels.extend(labels_batch)

        # compute all metrics on this batch
        summary_batch = {metric: metrics[metric](output_batch, labels_batch)
                         for metric in metrics}
        summary_batch['loss'] = loss.item()
        summ.append(summary_batch)

    # compute mean of all metrics in summary
    metrics_name = {metric: np.mean([x[metric]
                                     for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v)
                                for k, v in metrics_name.items())
    logging.info("- {} Metrics : ".format(which_set) + metrics_string)

    compute_and_save_f1(saved_outputs, saved_labels, file)

    return metrics_name


if __name__ == '__main__':
    """
        Evaluate the model on the train, validation, and test set.
    """
    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()     # use GPU is available

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda:
        torch.cuda.manual_seed(230)

    # Get the logger
    utils.set_logger(os.path.join(args.model_dir, 'trainAndValidation.log'))

    # Create the input data pipeline
    logging.info("Creating the dataset...")

    # fetch dataloaders
    dataloaders = data_loader.fetch_dataloader(['train', 'val', 'test'], args.data_dir, params)
    train_dl = dataloaders['train']
    val_dl = dataloaders['val']
    test_dl = dataloaders['test']

    logging.info("- done.")

    # Define the model
    model, loss_fn, metrics = get_model_loss_metrics(args, params)


    logging.info("Starting evaluation and calculation of F1 Scores")

    # Reload weights from the saved file
    utils.load_checkpoint(os.path.join(
        args.model_dir, args.restore_file + '.pth.tar'), model)

    # Evaluate Train
    confus_save_path = os.path.join(
        args.model_dir, "confus_f1_train_{}.json".format(args.restore_file))
    train_metrics = evaluate(model, loss_fn, train_dl, metrics, params, 'Train', confus_save_path, args)
    save_path = os.path.join(
        args.model_dir, "metrics_train_{}.json".format(args.restore_file))
    utils.save_dict_to_json(train_metrics, save_path)

    # Evaluate Validation
    confus_save_path = os.path.join(
        args.model_dir, "confus_f1_val_{}.json".format(args.restore_file))
    val_metrics = evaluate(model, loss_fn, val_dl, metrics, params, 'Val', confus_save_path, args)
    save_path = os.path.join(
        args.model_dir, "metrics_val_{}.json".format(args.restore_file))
    utils.save_dict_to_json(val_metrics, save_path)
