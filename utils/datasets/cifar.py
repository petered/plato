import os
from utils.datasets.datasets import DataSet, DataCollection
from utils.file_getter import get_file
import pickle
import numpy as np

__author__ = 'peter'


def get_cifar_10_dataset(n_training_samples = None, n_test_samples = None):
    """
    :return: The CIFAR-10 dataset, which consists of 50000 training and 10000 test images.
        Images are 32x32 uint8 RGB images (n_samples, 3, 32, 32) of 10 categories of objects.
        Targets are integer labels in the range [0, 9]
    """
    # TODO: Make method for downloading/unpacking data (from http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)
    # We have this for single files already, but in this case the gz contains a folder with the files in it.

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

    if n_training_samples is not None:
        x_tr = x_tr[:n_training_samples]
        y_tr = y_tr[:n_training_samples]
    if n_test_samples is not None:
        x_ts = x_ts[:n_test_samples]
        y_ts = y_ts[:n_test_samples]

    return DataSet(training_set=DataCollection(x_tr, y_tr), test_set=DataCollection(x_ts, y_ts), name = 'CIFAR-10')


if __name__ == '__main__':

    from plotting.easy_plotting import ezplot

    dataset = get_cifar_10_dataset()
    n_samples = 100

    ezplot({
        'sampled training images': np.swapaxes(dataset.training_set.input[:n_samples], 1, 3).reshape(10, 10, 32, 32, 3),
        'sampled training labels': dataset.training_set.target[:n_samples].reshape(10, 10)
        }, cmap = 'Paired')
