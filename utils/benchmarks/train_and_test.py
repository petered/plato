import logging
import numpy as np

__author__ = 'peter'

"""
All evaluation functions in here are of the form

score = evaluation_fcn(actual, target)

Where:
    score is a scalar
    actual is an (n_samples, ...) array
    target is an (n_samples, ....) array
"""


def train_online_predictor(predictor, training_set, minibatch_size, n_epochs = 1):
    """
    Train a predictor on the training set
    :param predictor: An IPredictor object
    :param training_set: A DataCollection object
    :param minibatch_size: An integer, or 'full' for full batch training
    :param n_epochs: Number of passes to make over the training set.
    """
    print 'Training Predictor %s...' % (predictor, )
    for (_, data, target) in training_set.minibatch_iterator(minibatch_size = minibatch_size, epochs = n_epochs, single_channel = True):
        predictor.train(data, target)
    print 'Done.'


def evaluate_predictor(predictor, test_set, evaluation_function):
    if isinstance(evaluation_function, str):
        evaluation_function = get_evaluation_function(evaluation_function)
    output = predictor.predict(test_set.input)
    score = evaluation_function(actual = output, target = test_set.target)
    return score


def get_evaluation_function(name):
    return {
        'mse': mean_squared_error,
        'mean_squared_error': mean_squared_error,
        'percent_argmax_correct': percent_argmax_correct,
        'percent_correct': percent_correct,
         }[name]


def mean_squared_error(actual, target):
    return np.mean(np.sum((actual-target)**2, axis = -1), axis = -1)


def fraction_correct(actual, target):
    return np.mean(actual == target)


def percent_correct(actual, target):
    return 100*fraction_correct(actual, target)


def percent_argmax_correct(actual, target):
    """
    :param actual: An (n_samples, n_dims) array
    :param target: An (n_samples, ) array of indices OR an (n_samples, n_dims) array
    :return:
    """
    actual = collapse_onehot_if_necessary(actual)
    target = collapse_onehot_if_necessary(target)
    return 100*fraction_correct(actual, target)


def collapse_onehot_if_necessary(output_data):
    """
    Given an input that could either be in onehot encoding or not, return it in onehot encoding.

    :param output_data: Either an (n_samples, n_dims) array, or an (n_samples, ) array of labels.
    :return: An (n_samples, ) array.
    """
    if output_data.ndim == 2:
        return np.argmax(output_data, axis = 1)
    else:
        assert output_data.ndim == 1 and output_data.dtype in (int, 'int32', bool)
        return output_data
