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
    collections = [DataCollection(data = data[subset][0], labels = data[subset][1]) for subset in ['training_set', 'test_set', 'validation_set']]
    return DataSet(*collections)
