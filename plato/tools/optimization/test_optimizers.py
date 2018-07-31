from plato.core import symbolic, add_update
from plato.tools.optimization.demo_compare_optimizers import get_experiments
from plato.tools.optimization.optimizers import GradientDescent, Adam, AdaMax, RMSProp, get_named_optimizer
from plato.tools.regressors.online_regressor import OnlineRegressor
from artemis.ml.predictors.predictor_tests import assert_online_predictor_not_broken
import theano.tensor as tt
import numpy as np


def _test_optimizer_on_simple_classification_problem(optimizer):

    # Logistic Regression
    assert_online_predictor_not_broken(
        predictor_constructor = lambda n_dim_in, n_dim_out:
            OnlineRegressor(
                input_size = n_dim_in,
                output_size=n_dim_out,
                optimizer=optimizer,
                regressor_type = 'logistic'
                ).compile(),
        categorical_target=False,
        n_epochs=20
        )


def test_gradient_descent_optimizer():
    _test_optimizer_on_simple_classification_problem(GradientDescent(eta=0.01))


def test_adam_optimizer():
    _test_optimizer_on_simple_classification_problem(Adam(alpha=0.01))


def test_adamax_optimizer():
    _test_optimizer_on_simple_classification_problem(AdaMax(alpha=0.01))


def test_unknown_shape():

    @symbolic
    def func(x, optimizer):
        loss = tt.sum((x-3)**2)
        updates = optimizer.get_updates(cost=loss, parameters=[x])
        for p, v in updates[1:]:
            add_update(p, v)
        return updates[0][1]

    x_base = np.random.RandomState(1234).randn(3, 4)
    for opt in ('adam', 'adamax', 'adagrad', 'rmsprop'):
        print('Running Optimizer: {}'.format(opt))
        optimizer = get_named_optimizer(opt, learning_rate=0.5)
        x = x_base
        f = func.partial(optimizer = optimizer).compile()
        for _ in range(50):
            x = f(x)
        error = np.abs(x-3)
        print('Mean Error: {}'.format(error.mean()))
        assert np.all(np.abs(x-3)<1.)


def test_demo_compare_optimizers():

    for exp_name, exp in get_experiments().iteritems():
        print('Running %s' % exp_name)
        exp()


if __name__ == '__main__':
    # test_gradient_descent_optimizer()
    # test_adam_optimizer()
    # test_adamax_optimizer()
    test_unknown_shape()

