from plato.tools.cost import negative_log_likelihood, percent_correct
from plato.tools.online_prediction.compare_symbolic_predictors import plot_records, CompareOnlinePredictors
from plato.tools.networks import MultiLayerPerceptron
from plato.tools.online_prediction.online_predictors import GradientBasedPredictor
from plato.tools.optimizers import SimpleGradientDescent, AdaMax
from utils.datasets.mnist import get_mnist_dataset
import numpy as np
from utils.tools.mymath import sqrtspace

__author__ = 'peter'


comparisons = lambda: None


"""
Here we run a comparison between SGD and Durk's new pet: AdaMax.  We run them both on MNIST for 50 epochs.
"""
comparisons.adamax_showdown = lambda test_mode = False: CompareOnlinePredictors(
    dataset = get_mnist_dataset(),
    classifier_constructors = {
        'sgd': lambda: GradientBasedPredictor(
            function = MultiLayerPerceptron(layer_sizes=[500, 10], input_size = 784, hidden_activation='sig', output_activation='lin', w_init_mag=0.01, rng = np.random.RandomState(5)),
            cost_function = negative_log_likelihood,
            optimizer = SimpleGradientDescent(eta = 0.1),
            ),
        'adamax': lambda: GradientBasedPredictor(
            function = MultiLayerPerceptron(layer_sizes=[500, 10], input_size = 784, hidden_activation='sig', output_activation='lin', w_init_mag=0.01, rng = np.random.RandomState(5)),
            cost_function = negative_log_likelihood,
            optimizer = AdaMax(alpha = 1e-3),
            ),
        },
    minibatch_size = {
        'sgd': 20,
        'adamax': 20,
        },
    test_points = sqrtspace(0, 0.1, 2) if test_mode else sqrtspace(0, 30, 50),
    evaluation_function = percent_correct
    )


if __name__ == '__main__':

    comparison = comparisons.adamax_showdown()
    records = comparison()
    plot_records(records)
