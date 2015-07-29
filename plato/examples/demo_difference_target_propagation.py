from general.test_mode import is_test_mode, set_test_mode
from plato.interfaces.decorators import set_enable_omniscence
from plato.tools.cost import mean_squared_error, softmax_negative_log_likelihood, softmax_mean_xe, mean_xe
from plato.tools.difference_target_prop import DifferenceTargetMLP, ReversedDifferenceTargetLayer, PerceptronLayer
from plato.tools.networks import normal_w_init, MultiLayerPerceptron
from plato.tools.online_prediction.online_predictors import GradientBasedPredictor
from plato.tools.optimizers import AdaMax
from plotting.db_plotting import dbplot
from plotting.matplotlib_backend import set_default_figure_size
from utils.benchmarks.plot_learning_curves import plot_learning_curves
from utils.benchmarks.predictor_comparison import compare_predictors
from utils.benchmarks.train_and_test import percent_argmax_correct
from utils.datasets.mnist import get_mnist_dataset
from utils.tools.mymath import sqrtspace
from utils.tools.processors import OneHotEncoding


__author__ = 'peter'


def demo_backprop_vs_difference_target_prop(
        hidden_sizes = [240],
        n_epochs = 10,
        minibatch_size = 20,
        n_tests = 20,
        predictors = ['backprop-MLP', 'DTP-MLP', 'RevDTP-MLP']
        ):
    """

    :param hidden_sizes:
    :param n_epochs:
    :param minibatch_size:
    :param n_tests:
    :return:
    """

    set_enable_omniscence(True)

    dataset = get_mnist_dataset(flat = True)
    dataset = dataset.process_with(targets_processor=lambda (x, ): (OneHotEncoding(10)(x).astype(int), ))

    if is_test_mode():
        dataset.shorten(200)
        n_epochs = 0.1
        n_tests = 3

    set_default_figure_size(12, 9)

    all_predictors = {
            'backprop-MLP': GradientBasedPredictor(
                function = MultiLayerPerceptron(
                    layer_sizes = hidden_sizes + [dataset.target_size],
                    input_size = dataset.input_size,
                    hidden_activation='tanh',
                    output_activation='sig',
                    w_init = normal_w_init(mag = 0.01, seed = 5)
                    ),
                cost_function = mean_squared_error,
                optimizer = AdaMax(0.01),
                ).compile(),
            'DTP-MLP': DifferenceTargetMLP.from_initializer(
                input_size = dataset.input_size,
                output_size = dataset.target_size,
                hidden_sizes = hidden_sizes,
                optimizer_constructor = lambda: AdaMax(0.01),
                input_activation='sigm',
                hidden_activation='tanh',
                output_activation='softmax',
                w_init_mag=0.01,
                noise = 1,
                ).compile(),
            'RevDTP-MLP': DifferenceTargetMLP.from_initializer(
                input_size = dataset.input_size,
                output_size = dataset.target_size,
                hidden_sizes = hidden_sizes,
                optimizer_constructor = lambda: AdaMax(0.01),
                input_activation='sigm',
                hidden_activation='tanh',
                output_activation='softmax',
                w_init_mag=0.01,
                noise = 1,
                layer_constructor = ReversedDifferenceTargetLayer.from_initializer
                ).compile(),
            'perceptron': DifferenceTargetMLP(
                layers=[PerceptronLayer.from_initializer(n_in, n_out, initial_mag=2)
                        for n_in, n_out in zip([dataset.input_size]+hidden_sizes, hidden_sizes+[dataset.target_size])],
                output_cost_function = None
                ).compile()
        }

    assert all(p in all_predictors for p in predictors), 'Not all predictors you listed: %s, exist' % (predictors, )

    return compare_predictors(
        dataset=dataset,
        online_predictors = {name: p for name, p in all_predictors.iteritems() if name in predictors},
        minibatch_size = minibatch_size,
        test_epochs = sqrtspace(0, n_epochs, n_tests),
        evaluation_function = percent_argmax_correct,
        accumulators='avg'
        )


def run_and_plot(training_scheme):
    learning_curves = training_scheme()
    plot_learning_curves(learning_curves)


EXPERIMENTS = dict()

EXPERIMENTS['backprop-vs-dtp'] = lambda: plot_learning_curves(demo_backprop_vs_difference_target_prop(predictors = ['DTP-MLP', 'backprop-MLP']))

EXPERIMENTS['DTP-vs-RevDTP'] = lambda: plot_learning_curves(demo_backprop_vs_difference_target_prop(predictors = ['DTP-MLP', 'RevDTP-MLP']))
"""
Result: The alternative DTP, where summation is done on the presigmoids, works the same or slightly better!
"""

EXPERIMENTS['perceptron'] = lambda: plot_learning_curves(demo_backprop_vs_difference_target_prop(
    predictors = ['perceptron'], hidden_sizes=[400, 200], n_epochs=60,
))


if __name__ == '__main__':

    which_experiment = 'perceptron'
    set_test_mode(False)

    EXPERIMENTS[which_experiment]()
