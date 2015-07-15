from fileman.experiment_record import register_experiment, run_experiment
from general.test_mode import is_test_mode, set_test_mode
from plato.interfaces.decorators import set_enable_omniscence
from plato.tools.cost import mean_squared_error, mean_xe
from plato.tools.difference_target_prop import DifferenceTargetMLP
from plato.tools.difference_target_prop_variations import ReversedDifferenceTargetLayer, PerceptronLayer
from plato.tools.networks import normal_w_init, MultiLayerPerceptron
from plato.tools.online_prediction.online_predictors import GradientBasedPredictor
from plato.tools.optimizers import AdaMax, SimpleGradientDescent, RMSProp, GradientDescent
from plotting.db_plotting import dbplot
from plotting.matplotlib_backend import set_default_figure_size
from utils.benchmarks.plot_learning_curves import plot_learning_curves
from utils.benchmarks.predictor_comparison import compare_predictors, assess_online_predictor
from utils.benchmarks.train_and_test import percent_argmax_correct
from utils.datasets.mnist import get_mnist_dataset
from utils.tools.mymath import sqrtspace
from utils.tools.processors import OneHotEncoding


__author__ = 'peter'


def demo_run_dtp_on_mnist(
        hidden_sizes = [240],
        n_epochs = 20,
        n_tests = 20,
        minibatch_size=100,
        input_activation = 'sigm',
        hidden_activation = 'tanh',
        output_activation = 'softmax',
        optimizer_constructor = lambda: RMSProp(0.001),
        noise = 1,
        ):

    dataset = get_mnist_dataset(flat = True).to_onehot()
    predictor = DifferenceTargetMLP.from_initializer(
            input_size = dataset.input_size,
            output_size = dataset.target_size,
            hidden_sizes = hidden_sizes,
            optimizer_constructor = optimizer_constructor,  # Note that RMSProp/AdaMax way outperform SGD here.
            input_activation=input_activation,
            hidden_activation=hidden_activation,
            output_activation=output_activation,
            w_init_mag=0.01,
            output_cost_function=None,
            noise = noise,
            ).compile()

    result = assess_online_predictor(
        predictor = predictor,
        dataset = dataset,
        minibatch_size=minibatch_size,
        evaluation_function='percent_argmax_correct',
        test_epochs = sqrtspace(0, n_epochs, n_tests),
        test_callback=lambda p: dbplot(p.symbolic_predictor.layers[0].w.get_value().T.reshape(-1, 28, 28))
        )

    plot_learning_curves(result)


def demo_perceptron_dtp(
        hidden_sizes = [240],
        n_epochs = 20,
        n_tests = 20,
        minibatch_size=100,
        ):
    dataset = get_mnist_dataset(flat = True).to_onehot()
    predictor = DifferenceTargetMLP(
        layers=[PerceptronLayer.from_initializer(n_in, n_out, initial_mag=2)
                for n_in, n_out in zip([dataset.input_size]+hidden_sizes, hidden_sizes+[dataset.target_size])],
        output_cost_function = None
        ).compile()

    result = assess_online_predictor(
        predictor = predictor,
        dataset = dataset,
        minibatch_size=minibatch_size,
        evaluation_function='percent_argmax_correct',
        test_epochs = sqrtspace(0, n_epochs, n_tests),
        )

    plot_learning_curves(result)


def demo_compare_dtp_optimizers(
        hidden_sizes = [240],
        n_epochs = 10,
        minibatch_size = 20,
        n_tests = 20,
        hidden_activation = 'tanh',
        ):

    dataset = get_mnist_dataset(flat = True).to_onehot()

    def make_dtp_net(optimizer_constructor, output_fcn):
        return DifferenceTargetMLP.from_initializer(
            input_size = dataset.input_size,
            output_size = dataset.target_size,
            hidden_sizes = hidden_sizes,
            optimizer_constructor = optimizer_constructor,
            input_activation='sigm',
            hidden_activation=hidden_activation,
            output_activation=output_fcn,
            w_init_mag=0.01,
            noise = 1,
            ).compile()

    learning_curves = compare_predictors(
        dataset=dataset,
        online_predictors = {
            'SGD-0.001': make_dtp_net(lambda: SimpleGradientDescent(0.001), output_fcn = 'softmax'),
            'AdaMax-0.001': make_dtp_net(lambda: AdaMax(0.001), output_fcn = 'softmax'),
            'RMSProp-0.001': make_dtp_net(lambda: RMSProp(0.001), output_fcn = 'softmax'),
            },
        minibatch_size = minibatch_size,
        test_epochs = sqrtspace(0, n_epochs, n_tests),
        evaluation_function = percent_argmax_correct,
        )

    plot_learning_curves(learning_curves)




def demo_compare_dtp_methods(
        hidden_sizes = [240],
        n_epochs = 10,
        minibatch_size = 20,
        n_tests = 20,
        hidden_activation = 'tanh',
        predictors = ['backprop-MLP', 'DTP-MLP', 'LinDTP-MLP']
        ):
    """
    ;

    :param hidden_sizes:
    :param n_epochs:
    :param minibatch_size:
    :param n_tests:
    :return:
    """

    set_enable_omniscence(True)

    dataset = get_mnist_dataset(flat = True, binarize = True)
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
                    hidden_activation=hidden_activation,
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
                hidden_activation=hidden_activation,
                output_activation='softmax',
                w_init_mag=0.01,
                noise = 1,
                ).compile(),
            'LinDTP-MLP': DifferenceTargetMLP.from_initializer(
                input_size = dataset.input_size,
                output_size = dataset.target_size,
                hidden_sizes = hidden_sizes,
                optimizer_constructor = lambda: AdaMax(0.01),
                input_activation='sigm',
                hidden_activation=hidden_activation,
                output_activation='softmax',
                w_init_mag=0.01,
                noise = 1,
                layer_constructor = ReversedDifferenceTargetLayer.from_initializer
                ).compile(),
        }

    assert all(p in all_predictors for p in predictors), 'Not all predictors you listed: %s, exist' % (predictors, )

    learning_curves = compare_predictors(
        dataset=dataset,
        online_predictors = {name: p for name, p in all_predictors.iteritems() if name in predictors},
        minibatch_size = minibatch_size,
        test_epochs = sqrtspace(0, n_epochs, n_tests),
        evaluation_function = percent_argmax_correct,
        # online_test_callbacks={'perceptron': lambda p: dbplot(p.symbolic_predictor.layers[0].w.get_value().T.reshape(-1, 28, 28))},
        # accumulators='avg'
        )

    plot_learning_curves(learning_curves)


def run_and_plot(training_scheme):
    learning_curves = training_scheme()
    plot_learning_curves(learning_curves)


register_experiment(
    name = 'standard-dtp',
    function = lambda: demo_run_dtp_on_mnist(),
    description="Train Difference Target Propagation on MNIST"
    )


register_experiment(
    name = 'all-relu-dtp',
    function = lambda: demo_run_dtp_on_mnist(
        input_activation='relu',
        hidden_activation='relu',
        output_activation='relu',
        optimizer_constructor=lambda: GradientDescent(eta = 0.01),
        ),
    description="DTP with an entirely RELU network: Works ok, with SGD, but then explodes."
    )

register_experiment(
    name = 'all-softplus-dtp',
    function = lambda: demo_run_dtp_on_mnist(
        input_activation='softplus',
        hidden_activation='softplus',
        output_activation='softplus',
        ),
    description="DTP with an entirely RELU network: Doesn't work that well."
    )

register_experiment(
    name = 'backprop-vs-dtp',
    function = lambda: demo_compare_dtp_methods(predictors = ['DTP-MLP', 'backprop-MLP']),
    description = "Compare Difference Target Propagation to ordinary Backpropagation"
    )

register_experiment(
    name = 'DTP-vs-LinDTP',
    function = lambda: demo_compare_dtp_methods(predictors = ['DTP-MLP', 'LinDTP-MLP']),
    description="See the results of doing the 'difference' calculation on the pre-sigmoid instead of post-sigmoid")


register_experiment(
    name = 'relu-dtp',
    function = lambda: demo_compare_dtp_methods(predictors = ['DTP-MLP', 'backprop-MLP', 'LinDTP-MLP'], hidden_activation='relu'),
    description = "Try Difference Target Prop (and LinDTP) with RELU units, see what happens."
    )

register_experiment(
    name = 'multi-level-perceptron',
    function = lambda: demo_perceptron_dtp(hidden_sizes=[400], n_epochs=60),
    description="Try Lin-DTP with sign-activation units and the perceptron learning rule (DTP just doesn't work, it seems)"
    )

register_experiment(
    name = 'compare_dtp_optimizers',
    function = lambda: demo_compare_dtp_optimizers(hidden_sizes=[400], n_epochs=20)
    )


if __name__ == '__main__':

    which_experiment = 'all-relu-dtp'
    set_test_mode(False)

    run_experiment(which_experiment)
