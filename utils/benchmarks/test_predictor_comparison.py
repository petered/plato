from sklearn.svm import SVC
from utils.benchmarks.predictor_comparison import compare_predictors
from utils.benchmarks.plot_learning_curves import plot_learning_curves
from utils.datasets.synthetic_clusters import get_synthetic_clusters_dataset
from utils.predictors.perceptron import Perceptron
import numpy as np
from utils.tools.mymath import sqrtspace

__author__ = 'peter'


def test_compare_predictors(plot = True):

    dataset = get_synthetic_clusters_dataset()

    w_constructor = lambda rng = np.random.RandomState(45): .1*rng.randn(dataset.input_shape[0], dataset.n_categories)
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
        evaluation_function='percent_correct'
        )

    assert 99 < records['SVM'].get_scores('Test') <= 100
    assert 20 < records['slow-perceptron'].get_scores('Test')[0] < 40 and 95 < records['slow-perceptron'].get_scores('Test')[-1] <= 100
    assert 20 < records['fast-perceptron'].get_scores('Test')[0] < 40 and 99 < records['fast-perceptron'].get_scores('Test')[-1] <= 100

    if plot:
        plot_learning_curves(records)


if __name__ == '__main__':
    test_compare_predictors()
