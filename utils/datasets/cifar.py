import os
from utils.datasets.datasets import DataSet, DataCollection
from utils.file_getter import get_file
import pickle
import numpy as np

__author__ = 'peter'


def get_cifar_10_dataset(n_training_samples = None, n_test_samples = None):
    """
    :return: A DataSet object containing the MNIST data
    """

    directory = 'data/cifar-10-batches-py'

    n_batches_to_read = 5 if n_training_samples is None else int(np.ceil(n_training_samples/10000.))

    file_paths = [get_file(os.path.join(os.path.join(directory, 'data_batch_%s' % (i, )))) for i in xrange(1, n_batches_to_read+1)] \
        + [get_file(os.path.join(os.path.join(directory, 'test_batch')))]

    data = []
    for file_path in file_paths:
        with open(file_path) as f:
            batch_data = pickle.load(f)
            data.append(batch_data)

    x_tr = np.concatenate([d['data'] for d in data[:-1]], axis = 0).reshape(-1, 3, 32, 32).swapaxes(2, 3)
    y_tr = np.concatenate([d['labels'] for d in data[:-1]], axis = 0)
    x_ts = data[-1]['data'].reshape(-1, 3, 32, 32).swapaxes(2, 3)
    y_ts = np.array(data[-1]['labels'])

    return DataSet(training_set=DataCollection(x_tr, y_tr), test_set=DataCollection(x_ts, y_ts), name = 'CIFAR-10')


if __name__ == '__main__':

    from plotting.easy_plotting import ezplot

    x_tr, y_tr, x_ts, y_ts = get_cifar_10_dataset().xyxy
    n_samples = 100

    ezplot({
        'sampled training images': np.swapaxes(x_tr[:n_samples], 1, 3).reshape(10, 10, 32, 32, 3),
        'sampled training labels': y_tr[:n_samples].reshape(10, 10)
        }, cmap = 'Paired')
