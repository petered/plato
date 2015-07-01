from fileman.experiment_record import run_experiment
from general.test_mode import is_test_mode, set_test_mode
from plato.interfaces.decorators import set_enable_omniscence
from plato.tools.cost import softmax_negative_log_likelihood, mean_squared_error
from plato.tools.difference_target_prop import DifferenceTargetMLP
from plato.tools.networks import MultiLayerPerceptron, normal_w_init
from plato.tools.online_prediction.online_predictors import GradientBasedPredictor
from plato.tools.optimizers import SimpleGradientDescent, AdaMax, AdaGrad
from plotting.db_plotting import dbplot
from plotting.matplotlib_backend import set_default_figure_size
from utils.benchmarks.plot_learning_curves import plot_learning_curves
from utils.benchmarks.predictor_comparison import compare_predictors
from utils.benchmarks.train_and_test import percent_argmax_correct
from utils.datasets.mnist import get_mnist_dataset
from utils.tools.mymath import sqrtspace
from utils.tools.processors import OneHotEncoding

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
    Compare mlps with different schemes for normalizing input.

    regular: Regular vanilla MLP
    normalize: Mean-subtract/normalize over minibatch
    normalize and scale: Mean-subtract/normalize over minibatch AND multiply by a trainable
        (per-unit) scale parameter.

    Conclusions: No significant benefit to scale parameter.  Normalizing gives
    a head start but incurs a small cost later on.  But really all classifiers are quite similar.

    :param hidden_size: Size of hidden layer
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


def backprop_vs_difference_target_prop(
        hidden_sizes = [240],
        n_epochs = 10,
        minibatch_size = 20,
        n_tests = 20
        ):

    set_enable_omniscence(True)

    dataset = get_mnist_dataset(flat = True)
    dataset = dataset.process_with(targets_processor=lambda (x, ): (OneHotEncoding(10)(x).astype(int), ))

    if is_test_mode():
        dataset.shorten(200)
        n_epochs = 0.1
        n_tests = 3

    set_default_figure_size(12, 9)

    return compare_predictors(
        dataset=dataset,
        online_predictors = {
            # 'backprop-mlp': GradientBasedPredictor(
            #     function = MultiLayerPerceptron(
            #         layer_sizes = hidden_sizes + [dataset.target_size],
            #         input_size = dataset.input_size,
            #         hidden_activation='tanh',
            #         output_activation='sig',
            #         w_init = normal_w_init(mag = 0.01, seed = 5)
            #         ),
            #     cost_function = mean_squared_error,
            #     optimizer = AdaMax(0.01),
            #     ).compile(),
            'difference-target-prop-mlp': DifferenceTargetMLP.from_initializer(
                input_size = dataset.input_size,
                output_size = dataset.target_size,
                hidden_sizes = hidden_sizes,
                optimizer_constructor = lambda: AdaMax(0.01),
                input_activation='sigm',
                hidden_activation='tanh',
                output_activation='softmax',
                w_init_mag=0.01,
                noise = 1,
            ).compile()
            },
        minibatch_size = minibatch_size,
        test_epochs = sqrtspace(0, n_epochs, n_tests),
        evaluation_function = percent_argmax_correct,
        )


def run_and_plot(training_scheme):
    learning_curves = training_scheme()
    plot_learning_curves(learning_curves)


def get_experiments():
    training_schemes = {
        'adamax-showdown': mnist_adamax_showdown,
        'mlp-normalization': mlp_normalization,
        'backprop-vs-dtp': backprop_vs_difference_target_prop,
        }
    experiments = {name: lambda sc=scheme: run_and_plot(sc) for name, scheme in training_schemes.iteritems()}
    return experiments


if __name__ == '__main__':

    test_mode = False
    experiment = 'backprop-vs-dtp'

    set_test_mode(test_mode)
    run_experiment(experiment, exp_dict=get_experiments(), show_figs = None, print_to_console=True)
