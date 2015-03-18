from experimental.sampling_mlp import GibbsSamplingMLP
from general.should_be_builtins import bad_value
from plato.tools.cost import negative_log_likelihood
from plato.tools.networks import MultiLayerPerceptron
from plato.tools.online_prediction.online_predictors import GradientBasedPredictor
from plato.tools.optimizers import SimpleGradientDescent
from utils.benchmarks.plot_learning_curves import plot_learning_curves
from utils.benchmarks.predictor_comparison import compare_predictors
from utils.datasets.mnist import get_mnist_dataset
from utils.datasets.synthetic_clusters import get_synthetic_clusters_dataset
from utils.tools.mymath import sqrtspace
import numpy as np


def demo_compare_sampling_predictors(which_dataset = 'mnist', test_mode = False):

    dataset = \
        get_synthetic_clusters_dataset(n_dims=100) if which_dataset == 'clusters' else \
        get_mnist_dataset(flat = True) if which_dataset == 'mnist' else \
        bad_value(which_dataset, 'No dataset named "%s"' % which_dataset)

    n_epochs = 10
    n_test_points = 20

    if test_mode:
        dataset.shorten(100)
        n_epochs = 1
        n_test_points = 3

    results = compare_predictors(
        dataset = dataset,
        online_predictors={
            'MLP': GradientBasedPredictor(
                function = MultiLayerPerceptron(layer_sizes = [100, dataset.n_categories], input_size = dataset.input_shape[0], output_activation='softmax', w_init = lambda n_in, n_out: 0.1*np.random.randn(n_in, n_out)),
                cost_function=negative_log_likelihood,
                optimizer=SimpleGradientDescent(eta = 0.1),
                ).compile(),
            'Gibbs-MLP': GibbsSamplingMLP(
                layer_sizes = [100, dataset.n_categories],
                input_size = dataset.input_shape[0],
                possible_ws=(-1, 0, 1),
                frac_to_update = .01,
                output_activation='softmax'
                ).compile(mode = 'tr'),
            },
        evaluation_function='percent_argmax_correct',
        minibatch_size=20,
        accumulators={
            'MLP': None,
            'Gibbs-MLP': 'avg',
            },
        test_epochs=sqrtspace(0, n_epochs, n_test_points),
        )
    plot_learning_curves(results)


if __name__ == '__main__':

    demo_compare_sampling_predictors()
