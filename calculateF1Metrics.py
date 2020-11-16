"""Evaluates the model"""

import argparse
import logging
import os

import numpy as np
import torch
from torch.autograd import Variable
import utils
import model.data_loader as data_loader
import regression_loss_and_metrics
import loss_and_metrics
import model.regression_adopted_cnn as regression_cnn

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir',
                    help="Directory containing params.json")
parser.add_argument('--restore_file', default='best', help="name of the file in --model_dir \
                     containing weights to load")
parser.add_argument('--testSet', default='False',
                    help="Indicate whether we should get the metrics for the test set.")
parser.add_argument('--trainAndVal', default='True',
                    help="Indicate whether we should get the metrics for the test set.")


def get_model_loss_metrics(args, params):
    if 'regression' in args.model_dir:
        model = regression_cnn.Regression_Adopted_NN(params).cuda() if params.cuda else regression_cnn.Regression_Adopted_NN(params)
        loss_fn = regression_loss_and_metrics.regression_loss_fn
        metrics = regression_loss_and_metrics.regression_metrics
        return model, loss_fn, metrics
    else:
        model = utils.get_desired_model(params)
        loss_fn = loss_and_metrics.loss_fn
        metrics = loss_and_metrics.metrics
        return model, loss_fn, metrics


def compute_and_save_f1(saved_outputs, saved_labels, file):
    conf_matrix, report = utils.f1_metrics(saved_outputs, saved_labels)

    text_file = open(file, "wt")
    text_file.write('Confusion matrix: \n {}\n\n Classification Report: \n {}'.format(conf_matrix, report))
    text_file.close()

def process_output(args, output_batch):
    if 'regression' not in args.model_dir:
        return np.argmax(output_batch, axis=1)
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


def get_data_dir():
    if 'regression' in args.model_dir:
        return 'regression/multiclass'
    if 'six_classes' in args.model_dir:
        return 'just_splitted/multiclass'
    if 'three_classes' in args.model_dir:
        return 'three_classes/multiclass'

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
    data_dir = get_data_dir()
    dataloaders = data_loader.fetch_dataloader(['train', 'val', 'test'], data_dir, params)
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
    if args.trainAndVal == "True":
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

    if args.testSet == "True":
        confus_save_path = os.path.join(
            args.model_dir, "confus_f1_test_{}.json".format(args.restore_file))
        test_metrics = evaluate(model, loss_fn, test_dl, metrics, params, 'Test', confus_save_path, args)
        save_path = os.path.join(
            args.model_dir, "metrics_test_{}.json".format(args.restore_file))
        utils.save_dict_to_json(test_metrics, save_path)
