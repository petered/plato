from utils.benchmarks.predictor_comparison import compare_predictors
from utils.benchmarks.plot_learning_curves import plot_learning_curves
from utils.benchmarks.compare_predictors import compare_predictors_old
from utils.datasets.datasets import DataSet, DataCollection
from utils.datasets.synthetic_logistic import get_logistic_regression_data
from utils.predictors.bad_predictors import MockPredictor
from utils.predictors.perceptron import Perceptron
import numpy as np
from utils.tools.mymath import sqrtspace, sigm
from utils.tools.processors import OneHotEncoding

__author__ = 'peter'


def test_compare_predictors_old():

    x_tr, y_tr, x_ts, y_ts, w_true = get_logistic_regression_data(noise_factor = 0.1)
    dataset = DataSet(DataCollection(x_tr, y_tr), DataCollection(x_ts, y_ts)).process_with(targets_processor=lambda (x, ): (OneHotEncoding()(x[:, 0]), ))

    w_init = 0.1*np.random.randn(dataset.training_set.input.shape[1], dataset.training_set.target.shape[1])
    records = compare_predictors_old(
        dataset = dataset,
        offline_predictor_constructors={
            'Optimal': lambda: MockPredictor(lambda x: sigm(x.dot(w_true)))
            },
        online_predictor_constructors={
            'fast-perceptron': lambda: Perceptron(alpha = 0.1, w = w_init.copy()),
            'slow-perceptron': lambda: Perceptron(alpha = 0.001, w = w_init.copy())
            },
        minibatch_size = 10,
        test_points = sqrtspace(0, 10, 20),
        evaluation_function='mse'
        )
    plot_learning_curves(records, hang = False)


if __name__ == '__main__':
    test_compare_predictors_old()
