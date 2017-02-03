import numpy as np
from artemis.experiments.experiment_record import run_experiment
from artemis.experiments.deprecated import register_experiment
from artemis.general.mymath import sqrtspace
from artemis.general.test_mode import is_test_mode
from artemis.ml.datasets.mnist import get_mnist_dataset
from artemis.ml.predictors.learning_curve_plots import plot_learning_curves
from artemis.ml.predictors.predictor_comparison import assess_online_predictor, compare_predictors
from artemis.ml.predictors.train_and_test import percent_argmax_correct
from artemis.plotting.db_plotting import dbplot
from plato.tools.common.bureaucracy import multichannel
from plato.tools.dtp.difference_target_prop import DifferenceTargetMLP, DifferenceTargetLayer
from plato.tools.dtp.difference_target_prop_variations import PerceptronLayer, PreActivationDifferenceTargetLayer
from plato.tools.optimization.cost import mean_squared_error, mean_abs_error
from plato.tools.optimization.optimizers import SimpleGradientDescent, AdaMax, RMSProp, GradientDescent

__author__ = 'peter'


def demo_compare_dtp_methods(
        predictor_constructors,
        n_epochs = 10,
        minibatch_size = 20,
        n_tests = 20,
        onehot = True,
        accumulator = None
        ):
    dataset = get_mnist_dataset(flat = True, binarize = False)
    n_categories = dataset.n_categories
    if onehot:
        dataset = dataset.to_onehot()

    if is_test_mode():
        dataset = dataset.shorten(200)
        n_epochs = 1
        n_tests = 2

    learning_curves = compare_predictors(
        dataset=dataset,
        online_predictors = {name: p(dataset.input_size, n_categories) for name, p in predictor_constructors.iteritems() if name in predictor_constructors},
        minibatch_size = minibatch_size,
        test_epochs = sqrtspace(0, n_epochs, n_tests),
        evaluation_function = percent_argmax_correct,
        # online_test_callbacks={'perceptron': lambda p: dbplot(p.symbolic_predictor.layers[0].w.get_value().T.reshape(-1, 28, 28))},
        accumulators=accumulator
        )

    plot_learning_curves(learning_curves)


def demo_compare_dtp_optimizers(
        hidden_sizes = [240],
        n_epochs = 10,
        minibatch_size = 20,
        n_tests = 20,
        hidden_activation = 'tanh',
        ):

    dataset = get_mnist_dataset(flat = True).to_onehot()

    if is_test_mode():
        dataset = dataset.shorten(200)
        n_epochs = 1
        n_tests = 2

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
            'SGD-0.001-softmax': make_dtp_net(lambda: SimpleGradientDescent(0.001), output_fcn = 'softmax'),
            'AdaMax-0.001-softmax': make_dtp_net(lambda: AdaMax(0.001), output_fcn = 'softmax'),
            'RMSProp-0.001-softmax': make_dtp_net(lambda: RMSProp(0.001), output_fcn = 'softmax'),
            'SGD-0.001-sigm': make_dtp_net(lambda: SimpleGradientDescent(0.001), output_fcn = 'sigm'),
            'AdaMax-0.001-sigm': make_dtp_net(lambda: AdaMax(0.001), output_fcn = 'sigm'),
            'RMSProp-0.001-sigm': make_dtp_net(lambda: RMSProp(0.001), output_fcn = 'sigm'),
            },
        minibatch_size = minibatch_size,
        test_epochs = sqrtspace(0, n_epochs, n_tests),
        evaluation_function = percent_argmax_correct,
        )

    plot_learning_curves(learning_curves)


def demo_perceptron_dtp(
        hidden_sizes = [240],
        n_epochs = 20,
        n_tests = 20,
        minibatch_size=100,
        lin_dtp = True,
        ):

    dataset = get_mnist_dataset(flat = True).to_onehot()

    if is_test_mode():
        dataset = dataset.shorten(200)
        n_epochs = 1
        n_tests = 2

    predictor = DifferenceTargetMLP(
        layers=[PerceptronLayer.from_initializer(n_in, n_out, initial_mag=2, lin_dtp = lin_dtp)
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


def demo_run_dtp_on_mnist(
        hidden_sizes = [240],
        n_epochs = 20,
        n_tests = 20,
        minibatch_size=100,
        input_activation = 'sigm',
        hidden_activation = 'tanh',
        output_activation = 'softmax',
        optimizer_constructor = lambda: RMSProp(0.001),
        normalize_inputs = False,
        local_cost_function = mean_squared_error,
        output_cost_function = None,
        noise = 1,
        lin_dtp = False,
        seed = 1234
        ):

    dataset = get_mnist_dataset(flat = True).to_onehot()
    if normalize_inputs:
        dataset = dataset.process_with(targets_processor=multichannel(lambda x: x/np.sum(x, axis = 1, keepdims=True)))
    if is_test_mode():
        dataset = dataset.shorten(200)
        n_epochs = 1
        n_tests = 2

    predictor = DifferenceTargetMLP.from_initializer(
            input_size = dataset.input_size,
            output_size = dataset.target_size,
            hidden_sizes = hidden_sizes,
            optimizer_constructor = optimizer_constructor,  # Note that RMSProp/AdaMax way outperform SGD here.
            # input_activation=input_activation,
            hidden_activation=hidden_activation,
            output_activation=output_activation,
            w_init_mag=0.01,
            output_cost_function=output_cost_function,
            noise = noise,
            cost_function = local_cost_function,
            layer_constructor=DifferenceTargetLayer.from_initializer if not lin_dtp else PreActivationDifferenceTargetLayer.from_initializer,
            rng = seed
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



def make_multi_level_perceptron(input_size, output_size, hidden_sizes, lin_dtp, initial_mag=2):
    return DifferenceTargetMLP(
        layers=[PerceptronLayer.from_initializer(n_in, n_out, initial_mag=initial_mag, lin_dtp = lin_dtp)
                for n_in, n_out in zip([input_size]+hidden_sizes, hidden_sizes+[output_size])],
        output_cost_function = None
        ).compile()

register_experiment(
    name = 'single-level-perceptron-DTP',
    function = lambda: demo_perceptron_dtp(hidden_sizes=[], n_epochs=60),
    description="Just to verify we're not crazy.  This should be equiv to a single layer perceptron (without biases though)",
    conclusion="Hovers kind of noisily just below 85%, as expected."
    )

register_experiment(
    name = 'multi-level-perceptron-DTP',
    function = lambda: demo_perceptron_dtp(hidden_sizes=[400], n_epochs=60, lin_dtp=False),
    description="Try DTP with one hidden layer using sign-activation units and the perceptron learning rule",
    conclusion="Doesn't work at all at all."
    )

register_experiment(
    name = 'multi-level-perceptron-LinDTP',
    function = lambda: demo_perceptron_dtp(hidden_sizes=[400], n_epochs=60, lin_dtp=True),
    description="Try Lin-DTP with sign-activation units and the perceptron learning rule (DTP just doesn't work, it seems)",
    conclusion="It can make it up to about 92.5% but then doesn't get any better (and can get worse)"
    )

register_experiment(
    name = 'deep-multi-level-perceptron-LinDTP',
    function = lambda: demo_perceptron_dtp(hidden_sizes=[400, 400], n_epochs=60, lin_dtp=True),
    description="Try the Lin-DTP perceptron with two hidden layers.",
    conclusion="Doesn't work at all"
    )


register_experiment(
    name = 'compare_multi_level_perceptron_dtp',
    function = lambda: demo_compare_dtp_methods(
        predictor_constructors={
            'perceptron': lambda n_in, n_out: make_multi_level_perceptron(n_in, n_out, hidden_sizes=[], lin_dtp=False),
            'multi-level-perceptron-DTP': lambda n_in, n_out: make_multi_level_perceptron(n_in, n_out, hidden_sizes=[400], lin_dtp=False),
            'multi-level-perceptron-LinDTP': lambda n_in, n_out: make_multi_level_perceptron(n_in, n_out, hidden_sizes=[400], lin_dtp=True),
            'deep-multi-level-perceptron-LinDTP': lambda n_in, n_out: make_multi_level_perceptron(n_in, n_out, hidden_sizes=[400, 400], lin_dtp=True),
            }
        ),
    description='Try various parameterizations of the "Multi-Level-Perceptron", trained with Difference Target Prop.  See what works.',
    conclusion='DTP fails on when used naively with perceptron activation functions.  LinDTP works (sort of, gets up to '
               '93% and kind of hovers), when there is only one hidden layer - it fails when there are more.   '
    )

register_experiment(
    name = 'compare_multi_level_perceptron_dtp_avg',
    function = lambda: demo_compare_dtp_methods(
        predictor_constructors={
            'perceptron': lambda n_in, n_out: make_multi_level_perceptron(n_in, n_out, hidden_sizes=[], lin_dtp=False),
            'multi-level-perceptron-DTP': lambda n_in, n_out: make_multi_level_perceptron(n_in, n_out, hidden_sizes=[400], lin_dtp=False),
            'multi-level-perceptron-LinDTP': lambda n_in, n_out: make_multi_level_perceptron(n_in, n_out, hidden_sizes=[400], lin_dtp=True),
            'deep-multi-level-perceptron-LinDTP': lambda n_in, n_out: make_multi_level_perceptron(n_in, n_out, hidden_sizes=[400, 400], lin_dtp=True),
            },
        accumulator='avg'
        ),
    description='Try various parameterizations of the "Multi-Level-Perceptron", trained with Difference Target Prop.  See what works.',
    conclusion='DTP fails on when used naively with perceptron activation functions.  LinDTP works (sort of, gets up to '
               '93% and kind of hovers), when there is only one hidden layer - it fails when there are more.   '
    )


register_experiment(
    name = 'side_by_side-multi-level-perceptron-dtp',
    function = lambda: demo_compare_dtp_methods(
        predictor_constructors={
            'perceptron': lambda n_in, n_out: make_multi_level_perceptron(n_in, n_out, hidden_sizes=[], lin_dtp=False),
            'multi-level-perceptron-DTP': lambda n_in, n_out: make_multi_level_perceptron(n_in, n_out, hidden_sizes=[400], lin_dtp=False),
            'multi-level-perceptron-LinDTP': lambda n_in, n_out: make_multi_level_perceptron(n_in, n_out, hidden_sizes=[400], lin_dtp=True),
            'deep-multi-level-perceptron-LinDTP': lambda n_in, n_out: make_multi_level_perceptron(n_in, n_out, hidden_sizes=[400, 400], lin_dtp=True),
            }
        ),
    description='Try various parameterizations of the "Multi-Level-Perceptron", trained with Difference Target Prop.  See what works.',
    conclusion='DTP fails on when used naively with perceptron activation functions.  LinDTP works (sort of, gets up to '
               '93% and kind of hovers), when there is only one hidden layer - it fails when there are more.   '
    )

register_experiment(
    name = 'side_by_side-multi-level-perceptron-dtp-avg',
    function = lambda: demo_compare_dtp_methods(
        predictor_constructors={
            'perceptron': lambda n_in, n_out: make_multi_level_perceptron(n_in, n_out, hidden_sizes=[], lin_dtp=False),
            'multi-level-perceptron-DTP': lambda n_in, n_out: make_multi_level_perceptron(n_in, n_out, hidden_sizes=[400], lin_dtp=False),
            'multi-level-perceptron-LinDTP': lambda n_in, n_out: make_multi_level_perceptron(n_in, n_out, hidden_sizes=[400], lin_dtp=True),
            'deep-multi-level-perceptron-LinDTP': lambda n_in, n_out: make_multi_level_perceptron(n_in, n_out, hidden_sizes=[400, 400], lin_dtp=True),
            },
        accumulator='avg'
        ),
    description='Try various parameterizations of the "Multi-Level-Perceptron", trained with Difference Target Prop.  See what works.',
    conclusion='DTP fails on when used naively with perceptron activation functions.  LinDTP works (sort of, gets up to '
               '93% and kind of hovers), when there is only one hidden layer - it fails when there are more.   '
    )


register_experiment(
    name = 'all-relu-dtp',
    function = lambda: demo_run_dtp_on_mnist(
        input_activation='relu',
        hidden_activation='relu',
        output_activation='relu',
        optimizer_constructor=lambda: GradientDescent(eta = 0.01),
        n_epochs=30,
        ),
    description="DTP with an entirely RELU network.",
    conclusion="Works pretty ok.  Unless it explodes.  Sometimes it doesn't explode.  Sometimes it does.  If it makes it past 93.5%, it generally survives."
    )


register_experiment(
    name = 'all-relu-dtp-exploding',
    function = lambda: demo_run_dtp_on_mnist(
        input_activation='relu',
        hidden_activation='relu',
        output_activation='relu',
        optimizer_constructor=lambda: GradientDescent(eta = 0.01),
        n_epochs=30,
        seed = 0
        ),
    description="DTP with an entirely RELU network.",
    conclusion="Works pretty ok.  Unless it explodes.  Sometimes it doesn't explode.  Sometimes it does.  If it makes it past 93.5%, it generally survives."
    )


register_experiment(
    name = 'all-relu-dtp-nonexploding',
    function = lambda: demo_run_dtp_on_mnist(
        input_activation='relu',
        hidden_activation='relu',
        output_activation='relu',
        optimizer_constructor=lambda: GradientDescent(eta = 0.01),
        n_epochs=30,
        seed = 1
        ),
    description="DTP with an entirely RELU network.",
    conclusion="Works pretty ok.  Unless it explodes.  Sometimes it doesn't explode.  Sometimes it does.  If it makes it past 93.5%, it generally survives."
    )


register_experiment(
    name = 'all-relu-dtp-abserror',
    function = lambda: demo_run_dtp_on_mnist(
        input_activation='relu',
        hidden_activation='relu',
        output_activation='relu',
        optimizer_constructor=lambda: GradientDescent(eta = 0.0001),
        local_cost_function=mean_abs_error,
        output_cost_function=mean_abs_error,
        n_epochs=30,
        ),
    description="Maybe L1 Error?",
    conclusion="Hells No.  Terrible idea."
    )


register_experiment(
    name = 'all-balanced-relu-dtp',
    function = lambda: demo_run_dtp_on_mnist(
        input_activation='relu',
        hidden_activation='balanced-relu',
        output_activation='relu',
        optimizer_constructor=lambda: GradientDescent(eta = 0.01),
        n_epochs=30,
        ),
    description="Try balanced RELU (where every second output is negated) as suggested by Xavier Glorot.  Don't see why "
        "this shouldn't make any difference, since inverting a RELU unit is like negating its outgoing weights, which are "
        "initialized randomly about zero.",
    conclusion="Doesn't help, as expected.."
    )

register_experiment(
    name = 'all-relu-dtp-momentum',
    function = lambda: demo_run_dtp_on_mnist(
        input_activation='relu',
        hidden_activation='relu',
        output_activation='relu',
        optimizer_constructor=lambda: GradientDescent(eta = 0.001, momentum=0.9),
        n_epochs=30,
        ),
    description="Lets see if momentum makes it more stable",
    conclusion="Not clear.  2/3 worked, 1/3 exploded."
    )

register_experiment(
    name = 'all-relu-dtp-rmsprop',
    function = lambda: demo_run_dtp_on_mnist(
        input_activation='relu',
        hidden_activation='relu',
        output_activation='relu',
        optimizer_constructor=lambda: RMSProp(learning_rate = 0.001),
        ),
    description="DTP with an entirely RELU network, using RMSprop as an optimizer",
    conclusion="RMSProp and RELU do not mix at all!"
    )


register_experiment(
    name = 'all-relu-dtp-adamax',
    function = lambda: demo_run_dtp_on_mnist(
        input_activation='relu',
        hidden_activation='relu',
        output_activation='relu',
        optimizer_constructor=lambda: AdaMax(alpha = 0.001),
        ),
    description="DTP with an entirely RELU network, using RMSprop as an optimizer",
    conclusion="AdaMax and RELU do not mix well either!  (not as horrible as RMSProp though)"
    )


register_experiment(
    name = 'all-relu-LinDTP',
    function = lambda: demo_run_dtp_on_mnist(
        input_activation='relu',
        hidden_activation='relu',
        output_activation='relu',
        optimizer_constructor=lambda: GradientDescent(eta = 0.01),
        n_epochs=30,
        lin_dtp = True
        ),
    description="DTP with an entirely RELU network.",
    conclusion="Works ok for a bit, then explodes and doesn't work at all.  Wait, somethimes it doesn't explode."
    )

register_experiment(
    name = 'all-relu-dtp-noiseless',
    function = lambda: demo_run_dtp_on_mnist(
        input_activation='relu',
        hidden_activation='relu',
        output_activation='relu',
        optimizer_constructor=lambda: GradientDescent(eta = 0.01),
        n_epochs=30,
        noise = 0
        ),
    description="See the effect of noise on RELU-DTP (in this architecture, it doesn't have a big effect on tanh-DTP).  "
        "Compare results to all-relu-dtp",
    conclusion="It seems that noise is really important.  The noiseless version is more likely to explode and seems to learn more slowly. "
    )

register_experiment(
    name = 'all-relu-dtp-minibatch1',
    function = lambda: demo_run_dtp_on_mnist(
        input_activation='relu',
        hidden_activation='relu',
        output_activation='relu',
        optimizer_constructor=lambda: GradientDescent(eta = 0.0001),
        n_epochs=30,
        minibatch_size=1
        ),
    description="RELU DTP with just 1 sample per minibatch.",
    conclusion="kaBOOM.  Unless you really lower the learning rate down to 0.0001.  In which case it's ok. "
        "Reached 94.17% in 6.73 epochs, which is when I lost patience."
    )

register_experiment(
    name = 'all-norm-relu-dtp',
    function = lambda: demo_run_dtp_on_mnist(
        input_activation='safenorm-relu',
        hidden_activation='safenorm-relu',
        output_activation='safenorm-relu',
        optimizer_constructor=lambda: SimpleGradientDescent(eta = 0.1),
        normalize_inputs=True,
        ),
    description="Now try with normalized-relu units",
    conclusion="Works, kind of, gets to like 93.5%.  Most hidden units seem to die.  At least it doesn't explode."
    )

register_experiment(
    name = 'all-softplus-dtp',
    function = lambda: demo_run_dtp_on_mnist(
        input_activation='softplus',
        hidden_activation='softplus',
        output_activation='softplus',
        optimizer_constructor=lambda: SimpleGradientDescent(eta = 0.01),
        ),
    description="DTP with an entirely softplus network.  It's known that RELUs have some problems as autoencoders, so we try softplus",
    conclusion = "Works badly for a while, and then explodes and doesn't work at all."
    )


"""
Other mlp done by changing code temporarily (and so not available here)

all-relu-dtp-nobias
We try removing biases from Difference Target propagation with RELU units.  This
causes the explosion to happen every time, and after achieving about 91% score.  We can
compensate by reducing the learning rate to 0.001, but then it takes forever to converge.
There's basically no middle ground - if you want a bearable learning rate, you get explosions.
"""


if __name__ == '__main__':
    run_experiment('all-softplus-dtp')
