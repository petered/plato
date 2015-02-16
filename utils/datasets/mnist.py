import gzip
from utils.datasets.datasets import DataSet, DataCollection
from utils.file_getter import get_file, unzip_gz
import pickle

__author__ = 'peter'


def get_mnist_dataset():
    """
    :return: A DataSet object containing the MNIST data
    """
    filename = get_file(
        local_name = 'data/mnist.pkl',
        url = 'http://deeplearning.net/data/mnist/mnist.pkl.gz',
        data_transformation = unzip_gz)

    with open(filename) as f:
        data = pickle.load(f)
    collections = [DataCollection(inputs = d[0], targets = d[1]) for d in data]
    return DataSet(*collections)
