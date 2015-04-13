from fileman.experiment_record import run_experiment
from general.test_mode import is_test_mode, set_test_mode
from plato.tools.cost import softmax_negative_log_likelihood
from plato.tools.networks import MultiLayerPerceptron, normal_w_init
from plato.tools.online_prediction.online_predictors import GradientBasedPredictor
from plato.tools.optimizers import SimpleGradientDescent, AdaMax
from utils.benchmarks.plot_learning_curves import plot_learning_curves
from utils.benchmarks.predictor_comparison import compare_predictors
from utils.benchmarks.train_and_test import percent_argmax_correct
from utils.datasets.mnist import get_mnist_dataset
from utils.tools.mymath import sqrtspace

__author__ = 'peter'


"""
Here we run a comparison between SGD and Durk's new pet: AdaMax.  We run them both on MNIST for 50 epochs.
"""


def mnist_adamax_showdown(hidden_size = 300, n_epochs = 10, n_tests = 20):

    dataset = get_mnist_dataset()

    if is_test_mode():
        dataset.shorten(200)
        n_epochs = 0.1
        n_tests = 3

    make_mlp = lambda optimizer: GradientBasedPredictor(
            function = MultiLayerPerceptron(
                layer_sizes=[hidden_size, dataset.n_categories],
                input_size = dataset.input_size,
                hidden_activation='sig',
                output_activation='lin',
                w_init = normal_w_init(mag = 0.01, seed = 5)
                ),
            cost_function = softmax_negative_log_likelihood,
            optimizer = optimizer,
            ).compile()

    return compare_predictors(
        dataset=dataset,
        online_predictors = {
            'sgd': make_mlp(SimpleGradientDescent(eta = 0.1)),
            'adamax': make_mlp(AdaMax(alpha = 1e-3)),
            },
        minibatch_size = 20,
        test_epochs = sqrtspace(0, n_epochs, n_tests),
        evaluation_function = percent_argmax_correct
        )


def mlp_normalization(hidden_size = 300, n_epochs = 30, n_tests = 50, minibatch_size=20):
    """

    :param hidden_size:
    :param n_epochs:
    :param n_tests:
    :param minibatch_size:
    :return:
    """
    dataset = get_mnist_dataset()

    if is_test_mode():
        dataset.shorten(200)
        n_epochs = 0.1
        n_tests = 3

    make_mlp = lambda normalize, scale: GradientBasedPredictor(
            function = MultiLayerPerceptron(
                layer_sizes=[hidden_size, dataset.n_categories],
                input_size = dataset.input_size,
                hidden_activation='sig',
                output_activation='lin',
                normalize_minibatch=normalize,
                scale_param=scale,
                w_init = normal_w_init(mag = 0.01, seed = 5)
                ),
            cost_function = softmax_negative_log_likelihood,
            optimizer = SimpleGradientDescent(eta = 0.1),
            ).compile()

    return compare_predictors(
        dataset=dataset,
        online_predictors = {
            'regular': make_mlp(normalize = False, scale = False),
            'normalize': make_mlp(normalize=True, scale = False),
            'normalize and scale': make_mlp(normalize=True, scale = True),
            },
        minibatch_size = minibatch_size,
        test_epochs = sqrtspace(0, n_epochs, n_tests),
        evaluation_function = percent_argmax_correct
        )


def run_and_plot(training_scheme):
    learning_curves = training_scheme()
    plot_learning_curves(learning_curves)


def get_experiments():
    training_schemes = {
        'adamax-showdown': mnist_adamax_showdown,
        'mlp-normalization': mlp_normalization
        }
    experiments = {name: lambda sc=scheme: run_and_plot(sc) for name, scheme in training_schemes.iteritems()}
    return experiments


if __name__ == '__main__':

    test_mode = False
    experiment = 'mlp-normalization'

    set_test_mode(test_mode)
    run_experiment(experiment, exp_dict=get_experiments(), show_figs = True, print_to_console=True)
