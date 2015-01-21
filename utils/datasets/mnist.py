from utils.datasets.datasets import DataSet, DataCollection
from utils.file_getter import get_file
import pickle

__author__ = 'peter'


def get_mnist_dataset():
    """
    :return: A DataSet object containing the MNIST data
    """
    filename = get_file('data/mnist.pkl')
    with open(filename) as f:
        data = pickle.load(f)
    collections = [DataCollection(inputs = d[0], targets = d[1]) for d in data]
    return DataSet(*collections)
