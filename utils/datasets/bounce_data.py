import numpy as np

__author__ = 'peter'


def get_bounce_data(period = 14, n_rounds = 1, onehot = False):
    """
    Data bounes between a max and min value.

    [0,1,2,3,2,1,0,0,1,2,3,2,1,0,...]

    :param period:
    :param n_rounds:
    :param onehot:
    :return:
    """

    n_samples = period * n_rounds

    x = np.arange(n_samples)

    x %= period
    x[x>=period/2] = period - x[x>=period/2]

    if onehot:
        onehot_x = np.zeros((n_samples, period/2+1))
        onehot_x[np.arange(n_samples), x] = 1
        return onehot_x
    else:
        return x
