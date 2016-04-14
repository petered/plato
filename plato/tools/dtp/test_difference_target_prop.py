from plato.tools.dtp.difference_target_prop import DifferenceTargetMLP
from plato.tools.optimization.optimizers import SimpleGradientDescent
from utils.predictors.predictor_tests import assert_online_predictor_not_broken
from plato.tools.dtp.demo_difference_target_propagation import ExperimentLibrary

__author__ = 'peter'


def test_difference_target_mlp():

    assert_online_predictor_not_broken(
        predictor_constructor=lambda n_in, n_out: DifferenceTargetMLP.from_initializer(
            input_size=n_in,
            output_size=n_out,
            hidden_sizes=[50],
            optimizer_constructor = lambda: SimpleGradientDescent(0.01),
            w_init_mag=0.01,
            rng = 1234
            ).compile(),
        minibatch_size=10,
        n_epochs=10,
        )


def test_demo_difference_target_prop():

    ExperimentLibrary.standard_dtp.test()
    ExperimentLibrary.linear_output_dtp.test()
    ExperimentLibrary.preact_dtp.test()


if __name__ == '__main__':

    test_demo_difference_target_prop()
    test_difference_target_mlp()
