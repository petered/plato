from plato.tools.difference_target_prop import DifferenceTargetMLP, DifferenceTargetLayer
from plato.tools.optimizers import SimpleGradientDescent
from utils.predictors.predictor_tests import assert_online_predictor_not_broken

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


if __name__ == '__main__':

    test_difference_target_mlp()
