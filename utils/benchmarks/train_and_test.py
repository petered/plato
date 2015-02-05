import logging
import numpy as np

__author__ = 'peter'


def train_online_predictor(predictor, training_set, iterator):
    logging.info('Training Predictor %s...' % (predictor, ))
    for (data, target) in iterator(training_set):
        predictor.train(data, target)
    logging.info('Done.')


def evaluate_predictor(predictor, test_set, evaluation_function):
    output = predictor.predict(test_set.input)
    score = evaluation_function(actual = output, target = test_set.target)
    return score


def get_evaluation_function(name):
    return {
        'mse': mean_squared_error,
        'mean_squared_error': mean_squared_error,
        'percent_argmax_correct': percent_argmax_correct
        }[name]


def mean_squared_error(actual, target):
    return np.mean(np.sum((actual-target)**2, axis = -1), axis = -1)


def fraction_correct(actual, target):
    return np.mean(np.eq(actual, target))


def percent_argmax_correct(actual, target):
    return 100*fraction_correct(actual, target)
