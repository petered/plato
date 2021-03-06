from collections import OrderedDict

from artemis.experiments.experiment_record import experiment_root
from artemis.experiments.ui import browse_experiments
from artemis.general.mymath import sqrtspace
from artemis.general.test_mode import is_test_mode
from artemis.ml.datasets.mnist import get_mnist_dataset
from artemis.ml.predictors.learning_curve_plots import plot_learning_curves
from artemis.ml.predictors.predictor_comparison import compare_predictors
from artemis.ml.predictors.train_and_test import percent_argmax_correct
from artemis.ml.tools.processors import OneHotEncoding
from artemis.plotting.pyplot_plus import set_default_figure_size
from plato.tools.common.online_predictors import GradientBasedPredictor
from plato.tools.dtp.difference_target_prop import DifferenceTargetMLP
from plato.tools.dtp.difference_target_prop_variations import PreActivationDifferenceTargetLayer, LinearDifferenceTargetMLP
from plato.tools.mlp.mlp import MultiLayerPerceptron
from plato.tools.optimization.cost import mean_squared_error
from plato.tools.optimization.optimizers import get_named_optimizer

__author__ = 'peter'


def get_predictor(predictor_type, input_size, target_size, hidden_sizes = [240], output_activation = 'sigm',
        hidden_activation = 'tanh', optimizer = 'adamax', learning_rate = 0.01, noise = 1, w_init=0.01,
        use_bias = True, rng = None):
    """
    Specify parameters that will allow you to construct a predictor

    :param predictor_type: String identifying the predictor class (see below)
    :param input_size: Integer size of the input vector.  Integer
    :param target_size: Integer size of the target vector
    :param hidden_sizes:
    :param input_activation:
    :param hidden_activation:
    :param optimizer:
    :param learning_rate:
    :return:
    """
    return {
        'MLP': lambda: GradientBasedPredictor(
            function = MultiLayerPerceptron.from_init(
                layer_sizes = [input_size] + hidden_sizes + [target_size],
                hidden_activations=hidden_activation,
                output_activation=output_activation,
                use_bias = use_bias,
                w_init = w_init,
                rng = rng
                ),
            cost_function = mean_squared_error,
            optimizer = get_named_optimizer(optimizer, learning_rate),
            ).compile(add_test_values = True),
        'DTP': lambda: DifferenceTargetMLP.from_initializer(
            input_size = input_size,
            output_size = target_size,
            hidden_sizes = hidden_sizes,
            optimizer_constructor = lambda: get_named_optimizer(optimizer, learning_rate),
            # input_activation=input_activation,
            hidden_activation=hidden_activation,
            output_activation=output_activation,
            w_init_mag=w_init,
            noise = noise,
            rng = rng,
            use_bias = use_bias,
            ).compile(add_test_values = True),
        'PreAct-DTP': lambda: DifferenceTargetMLP.from_initializer(
            input_size = input_size,
            output_size = target_size,
            hidden_sizes = hidden_sizes,
            optimizer_constructor = lambda: get_named_optimizer(optimizer, learning_rate),
            # input_activation=input_activation,
            hidden_activation=hidden_activation,
            output_activation=output_activation,
            w_init_mag=w_init,
            noise = noise,
            layer_constructor = PreActivationDifferenceTargetLayer.from_initializer,
            rng = rng,
            use_bias = use_bias,
            ).compile(add_test_values = True),
        'Linear-DTP': lambda: LinearDifferenceTargetMLP.from_initializer(
            input_size = input_size,
            output_size = target_size,
            hidden_sizes = hidden_sizes,
            optimizer_constructor = lambda: get_named_optimizer(optimizer, learning_rate),
            # input_activation=input_activation,
            hidden_activation=hidden_activation,
            output_activation='linear',
            w_init_mag=w_init,
            noise = noise,
            rng = rng,
            use_bias = use_bias,
            # layer_constructor = LinearDifferenceTargetLayer.from_initializer
            ).compile(add_test_values = True),
        }[predictor_type]()


@experiment_root
def demo_dtp_varieties(
        hidden_sizes = [240],
        n_epochs = 10,
        minibatch_size = 20,
        n_tests = 20,
        hidden_activation = 'tanh',
        output_activation = 'sigm',
        optimizer = 'adamax',
        learning_rate = 0.01,
        noise = 1,
        predictors = ['MLP', 'DTP', 'PreAct-DTP', 'Linear-DTP'],
        rng = 1234,
        use_bias = True,
        live_plot = False,
        plot = False
        ):
    """
    ;

    :param hidden_sizes:
    :param n_epochs:
    :param minibatch_size:
    :param n_tests:
    :return:
    """
    if isinstance(predictors, str):
        predictors = [predictors]

    dataset = get_mnist_dataset(flat = True)
    dataset = dataset.process_with(targets_processor=lambda (x, ): (OneHotEncoding(10)(x).astype(int), ))
    if is_test_mode():
        dataset = dataset.shorten(200)
        n_epochs = 0.1
        n_tests = 3

    set_default_figure_size(12, 9)

    predictors = OrderedDict((name, get_predictor(name, input_size = dataset.input_size, target_size=dataset.target_size,
            hidden_sizes=hidden_sizes, hidden_activation=hidden_activation, output_activation = output_activation,
            optimizer=optimizer, learning_rate=learning_rate, noise = noise, use_bias = use_bias, rng = rng)) for name in predictors)

    learning_curves = compare_predictors(
        dataset=dataset,
        online_predictors = predictors,
        minibatch_size = minibatch_size,
        test_epochs = sqrtspace(0, n_epochs, n_tests),
        evaluation_function = percent_argmax_correct,
        )

    if plot:
        plot_learning_curves(learning_curves)


def run_and_plot(training_scheme):
    learning_curves = training_scheme()
    plot_learning_curves(learning_curves)


# Train Difference Target Propagation on MNIST using standard settings, side_by_side to backprop.  This will be used as a baseline agains other mlp.
X = demo_dtp_varieties.add_root_variant('standard_dtp', predictors = ['MLP', 'DTP'])
X.add_variant(n_epochs=10)
X.add_variant(n_epochs=20)
# After 10 epochs:
#     MLP: 97.32
#     DTP: 96.61

# register_experiment('standard_dtp',
#     function = partial(demo_dtp_varieties, predictors = ['MLP', 'DTP']),
#     description="""Train Difference Target Propagation on MNIST using standard settings, side_by_side to backprop.  This will "
#         be used as a baseline agains other mlp.""",
#     versions = {'10_epoch': dict(n_epochs=10), '20_epoch': dict(n_epochs=20)},
#     current_version='10_epoch',
#     conclusion = """
#         After 10 epochs:
#             MLP: 97.32
#             DTP: 96.61
#
#         """
#     )


# Lets see change hidden units and try with and without bias.  In our basic spliking implementation,
# our units are like ReLUs with no bias... If relu with no-bias performs badly here, then maybe this is the reason
# that our spiking DTP is performing badly.
X=demo_dtp_varieties.add_variant('hidden_types', predictors = ['MLP', 'DTP'], n_epochs = 20)
X.add_variant('relu_withbias', hidden_activation='relu', use_bias=True, rng=0)
X.add_variant('tanh_withbias', hidden_activation='tanh', use_bias=True, rng=0)
X.add_variant('relu_sansbias', hidden_activation='relu', use_bias=False, rng=0)
X.add_variant('tanh_sansbias', hidden_activation='tanh', use_bias=False, rng=0)
X.add_variant('relu_slowlearn_withbias', hidden_activation='relu', use_bias=True, rng=0, learning_rate=0.002)
X.add_variant('relu_slowlearn_sansbias', hidden_activation='relu', use_bias=False, rng=0, learning_rate=0.002)
# learning_rate = 0.01 (default)
#
# relu_withbias
#     MLP: 98.00
#     DTP: 9.91
# tanh_withbias
#     MLP: 97.69
#     DTP: 96.48
# relu_sansbias
#     MLP: 98.02
#     DTP: 9.91 (kaboom)
# tanh_sansbias
#     MLP: 97.63
#     DTP: 96.68
# relu_slowlearn_withbias
#     DTP: 9.91
# relu_slowlearn_sansbias
#     DTP: 9.91
#
# Conclusion: ReLU units cause some instability in DTP.
# tanh always work reasonably well.


# register_experiment('hidden_types',
#     function = partial(demo_dtp_varieties, predictors = ['MLP', 'DTP'], n_epochs = 20),
#     description="""Lets see change hidden units and try with and without bias.  In our basic spliking implementation,
#         our units are like ReLUs with no bias... If relu with no-bias performs badly here, then maybe this is the reason
#         that our spiking DTP is performing badly.""",
#     versions = {
#         'relu_withbias': dict(hidden_activation='relu', use_bias=True, rng=0),
#         'tanh_withbias': dict(hidden_activation='tanh', use_bias=True, rng=0),
#         'relu_sansbias': dict(hidden_activation='relu', use_bias=False, rng=0),
#         'tanh_sansbias': dict(hidden_activation='tanh', use_bias=False, rng=0),
#         'relu_slowlearn_withbias': dict(hidden_activation='relu', use_bias=True, rng=0, learning_rate=0.002),
#         'relu_slowlearn_sansbias': dict(hidden_activation='relu', use_bias=False, rng=0, learning_rate=0.002),
#     },
#     current_version='relu_slowlearn_sansbias',
#     conclusion = """
#         learning_rate = 0.01 (default)
#
#         relu_withbias
#             MLP: 98.00
#             DTP: 9.91
#         tanh_withbias
#             MLP: 97.69
#             DTP: 96.48
#         relu_sansbias
#             MLP: 98.02
#             DTP: 9.91 (kaboom)
#         tanh_sansbias
#             MLP: 97.63
#             DTP: 96.68
#         relu_slowlearn_withbias
#             DTP: 9.91
#         relu_slowlearn_sansbias
#             DTP: 9.91
#
#         Conclusion: ReLU units cause some instability in DTP.
#         tanh always work reasonably well.
#         """
#     )


# See how linear output affects us.  This is mainly meant as a comparison to Linear DTP
X=demo_dtp_varieties.add_variant('linear_output_dtp', output_activation='linear', predictors = ['MLP', 'DTP'])
# MLP: 94.96
# DTP: 96.61

# register_experiment('linear_output_dtp',
#     function = lambda: demo_dtp_varieties(output_activation='linear', predictors = ['MLP', 'DTP']),
#     description="See how linear output affects us.  This is mainly meant as a comparison to Linear DTP",
#     conclusion = """
#         MLP: 94.96
#         DTP: 96.61
#     """
#     )


# See if noise is helpful (side_by_side to standard-dtp)
X=demo_dtp_varieties.add_variant('v', noise=0, predictors=['DTP'])
# DTP: 94.40
# So noise may be helpful but not critical.s

# register_experiment('standard_dtp_noiseless',
#     function = lambda: demo_dtp_varieties(noise=0, predictors=['DTP']),
#     description="See if noise is helpful (side_by_side to standard-dtp)",
#     conclusion = """
#         DTP: 94.40
#         So noise may be helpful but not critical.s
#         """
#     )

# Try doing the difference with the pre-activations
X=demo_dtp_varieties.add_variant('preact_dtp', predictors = 'PreAct-DTP')
# 96.64... woo

# register_experiment('preact_dtp',
#     function = partial(demo_dtp_varieties, predictors = 'PreAct-DTP'),
#     description="Try doing the difference with the pre-activations",
#     conclusion = """
#         96.64... woo@
#         """
#     )

# Try reversing the linearity and nonlinearity
X=demo_dtp_varieties.add_variant('linear_dtp', predictors = 'Linear-DTP')
X.add_variant('slowlearn', learning_rate=0.001)
X.add_variant('all_lin', hidden_activation='linear')
X.add_variant('relu', hidden_activation='relu')
X.add_variant('noiseless', noise=0)
# baseline:
# slowlearn:
# all_lin:
# relu: 87.97
# ... This is bad.  why does it do this?

# register_experiment('linear_dtp',
#     function = partial(demo_dtp_varieties, predictors = 'Linear-DTP'),
#     versions = {
#         'baseline': {},
#         'slowlearn': dict(learning_rate=0.001),
#         'all_lin': dict(hidden_activation='linear'),
#         'relu': dict(hidden_activation='relu'),
#         'noiseless': dict(noise=0),
#         },
#     current_version='noiseless',
#     description="Try reversing the linearity and nonlinearity",
#     conclusion = """
#         baseline:
#         slowlearn:
#         all_lin:
#         relu: 87.97
#
#         ... This is bad.  why does it do this?
#     """
#     )

# Now try with ReLU hidden units
X=demo_dtp_varieties.add_variant('relu_dtp', predictors = ['MLP', 'DTP'], hidden_activation = 'relu')
# MLP: 97.81
# DTP: 97.17

# register_experiment('relu_dtp',
#     function = partial(demo_dtp_varieties, predictors = ['MLP', 'DTP'], hidden_activation = 'relu'),
#     description = "Now try with ReLU hidden units",
#     conclusion = """
#         MLP: 97.81
#         DTP: 97.17
#     """
#     )


X=demo_dtp_varieties.add_variant('compare_dtp_optimizers', predictors = 'DTP')
X.add_variant('SGD_0.001_softmax', optimizer='sgd', learning_rate=0.001, output_activation='softmax')
X.add_variant('AdaMax_0.001_softmax', optimizer='adamax', learning_rate=0.001, output_activation='softmax')
X.add_variant('RMSProp_0.001_softmax', optimizer='rmsprop', learning_rate=0.001, output_activation='softmax')
X.add_variant('SGD_0.001_sigm', optimizer='sgd', learning_rate=0.001, output_activation='sigm')
X.add_variant('AdaMax_0.001_sigm', optimizer='adamax', learning_rate=0.001, output_activation='sigm')
X.add_variant('RMSProp_0.001_sigm', optimizer='rmsprop', learning_rate=0.001, output_activation='sigm')
# SGD converges much more slowly, at least at this learning rate.  AdaMax and RMSprop perform similarily.  "
#
# SGD_0.001_softmax:     72.69 -- learning rate slow... need to try for more epochs.
# AdaMax_0.001_softmax:  96.30
# RMSProp_0.001_softmax: 96.83
# SGD_0.001_sigm:        71.72 ???
# AdaMax_0.001_sigm:     96.34
# RMSProp_0.001_sigm:    96.35

if __name__ == '__main__':
    browse_experiments()
