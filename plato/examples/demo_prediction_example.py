from general.test_mode import is_test_mode
from plato.tools.optimization.cost import negative_log_likelihood_dangerous
from plato.tools.mlp.mlp import normal_w_init, MultiLayerPerceptron
from plato.tools.deprecated.old_mlp import OldMultiLayerPerceptron
from plato.tools.common.online_predictors import GradientBasedPredictor
from plato.tools.optimization.optimizers import SimpleGradientDescent
from sklearn.ensemble.forest import RandomForestClassifier
from utils.benchmarks.plot_learning_curves import plot_learning_curves
from utils.benchmarks.predictor_comparison import compare_predictors
from utils.benchmarks.train_and_test import percent_argmax_correct
from utils.datasets.mnist import get_mnist_dataset
import numpy as np
from utils.predictors.perceptron import Perceptron
from utils.tools.mymath import sqrtspace

__author__ = 'peter'


def compare_example_predictors(
        n_epochs = 5,
        n_tests = 20,
        minibatch_size = 10,
    ):
    """
    This demo shows how we can compare different online predictors.  The demo trains both predictors on the dataset,
    returning an object that contains the results.

    :param test_mode: Set this to True to just run the demo quicky (but not to completion) to see that it doesn't break.
    """

    dataset = get_mnist_dataset(flat = True)
    # "Flatten" the 28x28 inputs to a 784-d vector

    if is_test_mode():
        # Shorten the dataset so we run through it quickly in test mode.
        dataset = dataset.shorten(200)
        n_epochs = 1
        n_tests = 3

    # Here we compare three predictors on MNIST - an MLP, a Perceptron, and a Random Forest.
    # - The MLP is defined using Plato's interfaces - we create a Symbolic Predictor (GradientBasedPredictor) and
    #   then compile it into an IPredictor object
    # - The Perceptron directly implements the IPredictor interface.
    # - The Random Forest implements SciKit learn's predictor interface - that is, it has a fit(x, y) and a predict(x) method.
    learning_curve_data = compare_predictors(
        dataset = dataset,
        online_predictors = {
            'Perceptron': Perceptron(
                w = np.zeros((dataset.input_size, dataset.n_categories)),
                alpha = 0.001
                ).to_categorical(n_categories = dataset.n_categories),  # .to_categorical allows the perceptron to be trained on integer labels.
            'MLP': GradientBasedPredictor(
                function = MultiLayerPerceptron.from_init(
                    layer_sizes=[dataset.input_size, 500, dataset.n_categories],
                    hidden_activation='sig',  # Sigmoidal hidden units
                    output_activation='softmax',  # Softmax output unit, since we're doing multinomial classification
                    w_init = normal_w_init(mag = 0.01, seed = 5)
                ),
                cost_function = negative_log_likelihood_dangerous,  # "Dangerous" because it doesn't check to see that output is normalized, but we know it is because it comes from softmax.
                optimizer = SimpleGradientDescent(eta = 0.1),
                ).compile(),  # .compile() returns an IPredictor
            },
        offline_predictors={
            'RF': RandomForestClassifier(n_estimators = 40)
            },
        minibatch_size = minibatch_size,
        test_epochs = sqrtspace(0, n_epochs, n_tests),
        evaluation_function = percent_argmax_correct  # Compares one-hot
        )
    # Results is a LearningCurveData object
    return learning_curve_data


if __name__ == '__main__':

    records = compare_example_predictors(
        n_epochs=30,
        minibatch_size=20,
        test_mode = False,
        )
    plot_learning_curves(records)
