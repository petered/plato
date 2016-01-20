from plato.interfaces.helpers import SlowBatchNormalize
from plato.tools.common.online_predictors import GradientBasedPredictor
from plato.tools.mlp.modified_mlps import StatefulMultiLayerPerceptron
from plato.tools.optimization.optimizers import GradientDescent
from utils.predictors.predictor_tests import assert_online_predictor_not_broken

__author__ = 'peter'


def test_online_minibatch_normalization():

    assert_online_predictor_not_broken(
        predictor_constructor = lambda n_dim_in, n_dim_out:
            GradientBasedPredictor(
                function = StatefulMultiLayerPerceptron.from_init(
                    normalize_minibatch = SlowBatchNormalize(20),
                    output_activation = 'linear', hidden_activation = 'relu', layer_sizes=[n_dim_in, 50, n_dim_out], w_init_mag = 0.1, use_bias=False, rng=1234),
                cost_function = 'mse',
                optimizer=GradientDescent(eta=0.01)
                ).compile(),
        categorical_target=False,
        minibatch_size=1,
        n_epochs=3,
        initial_score_under = 50
        )


if __name__ == '__main__':
    test_online_minibatch_normalization()
