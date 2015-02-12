from plato.tools.online_prediction.online_predictors import GradientBasedClassifier
from plato.tools.online_prediction.compare_symbolic_predictors import CompareClassifiers, plot_records
from plato.tools.cost import PercentCorrect, NegativeLogLikelihood
from plato.tools.networks import MultiLayerPerceptron
from plato.tools.optimizers import SimpleGradientDescent, AdaMax
from utils.datasets.mnist import get_mnist_dataset
import numpy as np
from utils.tools.mymath import quadspace

__author__ = 'peter'


comparisons = lambda: None


comparisons.adamax_showdown = lambda: CompareClassifiers(
    dataset = get_mnist_dataset(),
    classifier_constructors = {
        'sgd': lambda: GradientBasedClassifier(
            function = MultiLayerPerceptron(layer_sizes=[500, 10], input_size = 784, hidden_activation='sig', output_activation='lin', w_init_mag=0.01, rng = np.random.RandomState(5)),
            cost_function = NegativeLogLikelihood(),
            optimizer = SimpleGradientDescent(eta = 0.1),
            ),
        'adamax': lambda: GradientBasedClassifier(
            function = MultiLayerPerceptron(layer_sizes=[500, 10], input_size = 784, hidden_activation='sig', output_activation='lin', w_init_mag=0.01, rng = np.random.RandomState(5)),
            cost_function = NegativeLogLikelihood(),
            optimizer = AdaMax(alpha = 1e-3),
            ),
        },
    minibatch_size = {
        'sgd': 20,
        'adamax': 20,
        },
    test_points = quadspace(0, 30, 50),
    evaluation_function = PercentCorrect()
    )


if __name__ == '__main__':

    comparison = comparisons.adamax_showdown()

    records = comparison()

    plot_records(records)