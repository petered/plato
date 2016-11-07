import pytest

from plato.interfaces.decorators import symbolic_updater
from plato.tools.common.online_predictors import GradientBasedPredictor
from plato.tools.mlp.demo_mnist_mlp import demo_mnist_mlp
from plato.tools.mlp.mlp import create_maxout_network, MultiLayerPerceptron
from plato.tools.optimization.cost import negative_log_likelihood_dangerous
from plato.tools.optimization.optimizers import SimpleGradientDescent
from artemis.ml.predictors.train_and_test import percent_argmax_correct
from artemis.ml.tools.iteration import zip_minibatch_iterate
from artemis.ml.datasets.synthetic_clusters import get_synthetic_clusters_dataset
from artemis.ml.predictors.predictor_tests import assert_online_predictor_not_broken


__author__ = 'peter'


def test_bare_bones_mlp(seed = 1234):
    """
    This verifies that the MLP works.  It's intentionally not using any wrappers on top of MLP to show its "bare bones"
    usage.  Wrapping in GradientBasedPredictor can simplify usage - see test_symbolic_predictors.
    """

    dataset = get_synthetic_clusters_dataset()

    mlp = MultiLayerPerceptron.from_init(
        layer_sizes = [dataset.input_size, 20, dataset.n_categories],
        hidden_activation = 'relu',
        output_activation = 'softmax',
        w_init = 0.01,
        rng = seed
        )

    fwd_fcn = mlp.compile()

    optimizer = SimpleGradientDescent(eta = 0.1)

    @symbolic_updater
    def train(x, y):
        output = mlp(x)
        cost = negative_log_likelihood_dangerous(output, y)
        optimizer(cost, mlp.parameters)

    train_fcn = train.compile()

    init_score = percent_argmax_correct(fwd_fcn(dataset.test_set.input), dataset.test_set.target)

    for x_m, y_m in zip_minibatch_iterate([dataset.training_set.input, dataset.training_set.target], minibatch_size=10, n_epochs=20):
        train_fcn(x_m, y_m)

    final_score = percent_argmax_correct(fwd_fcn(dataset.test_set.input), dataset.test_set.target)
    print 'Initial score: %s%%.  Final score: %s%%' % (init_score, final_score)
    assert init_score < 30
    assert final_score > 98


def test_mlp():

    assert_online_predictor_not_broken(
        predictor_constructor = lambda n_dim_in, n_dim_out:
            GradientBasedPredictor(
                function = MultiLayerPerceptron.from_init(
                    layer_sizes = [n_dim_in, 100, n_dim_out],
                    output_activation='softmax',
                    w_init = 0.1,
                    rng = 3252
                    ),
                cost_function=negative_log_likelihood_dangerous,
                optimizer=SimpleGradientDescent(eta = 0.1),
                ).compile(),
        categorical_target=True,
        minibatch_size=10,
        n_epochs=2
        )


def test_mlp_with_scale_learning():

    assert_online_predictor_not_broken(
        predictor_constructor = lambda n_dim_in, n_dim_out:
            GradientBasedPredictor(
                function = MultiLayerPerceptron.from_init(
                    layer_sizes = [n_dim_in, 100, n_dim_out],
                    output_activation='softmax',
                    scale_param = True,
                    w_init = 0.1,
                    rng = 3252
                    ),
                cost_function=negative_log_likelihood_dangerous,
                optimizer=SimpleGradientDescent(eta = 0.1),
                ).compile(),
        categorical_target=True,
        minibatch_size=10,
        n_epochs=2
        )


def test_maxout_mlp():

    assert_online_predictor_not_broken(
        predictor_constructor = lambda n_dim_in, n_dim_out:
            GradientBasedPredictor(
                function = create_maxout_network(
                    layer_sizes = [n_dim_in, 100, n_dim_out],
                    maxout_widths = 4,
                    output_activation = 'softmax',
                    w_init=0.01,
                    rng = 1234,
                ),
                cost_function=negative_log_likelihood_dangerous,
                optimizer=SimpleGradientDescent(eta = 0.1),
                ).compile(),
        categorical_target=True,
        minibatch_size=10,
        n_epochs=2
        )


def test_all_maxout_mlp():

    with pytest.raises(AssertionError):
        assert_online_predictor_not_broken(
            predictor_constructor = lambda n_dim_in, n_dim_out:
                GradientBasedPredictor(
                    function = create_maxout_network(
                        layer_sizes = [n_dim_in, 100, n_dim_out],
                        maxout_widths = 4,
                        output_activation = 'maxout',
                        w_init=0.01,
                    ),
                    cost_function=negative_log_likelihood_dangerous,
                    optimizer=SimpleGradientDescent(eta = 0.005),
                    ).compile(),
            categorical_target=True,
            minibatch_size=10,
            n_epochs=20
            )


if __name__ == '__main__':

    test_all_maxout_mlp()
    test_maxout_mlp()
    test_bare_bones_mlp()
    test_mlp()
    test_mlp_with_scale_learning()


def test_demo_mnist_mlp():
    demo_mnist_mlp()