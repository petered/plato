from sklearn.svm import SVC
from utils.benchmarks.predictor_comparison import compare_predictors
from utils.benchmarks.plot_learning_curves import plot_learning_curves
from utils.bureaucracy import multichannel
from utils.datasets.datasets import DataSet, DataCollection
from utils.datasets.synthetic_clusters import get_synthetic_clusters_dataset
from utils.datasets.synthetic_logistic import get_logistic_regression_data
from utils.predictors.bad_predictors import MockPredictor
from utils.predictors.perceptron import Perceptron
import numpy as np
from utils.tools.mymath import sqrtspace, sigm
from utils.tools.processors import OneHotEncoding

__author__ = 'peter'


def test_compare_predictors():

    # x_tr, y_tr, x_ts, y_ts = get_synthetic_clusters_dataset().xyxy
    # dataset = DataSet(DataCollection(x_tr, y_tr), DataCollection(x_ts, y_ts)).process_with(targets_processor=lambda (x, ): (OneHotEncoding()(x[:, 0]), ))


    dataset = get_synthetic_clusters_dataset() #.process_with(targets_processor=multichannel(OneHotEncoding()))

    w_constructor = lambda: 0.1*np.random.randn(dataset.training_set.input.shape[1], dataset.training_set.target.shape[1])
    records = compare_predictors(
        dataset = dataset,
        offline_predictors={
            'SVM': SVC()
            },
        online_predictors={
            'fast-perceptron': Perceptron(alpha = 0.1, w = w_constructor()).to_categorical(),
            'slow-perceptron': Perceptron(alpha = 0.001, w = w_constructor()).to_categorical()
            },
        minibatch_size = 10,
        test_epochs = sqrtspace(0, 10, 20),
        evaluation_function='percent_argmax_correct'
        )
    plot_learning_curves(records)


if __name__ == '__main__':
    test_compare_predictors()
