from plato.tools.optimization.optimizers import GradientDescent, Adam, AdaMax
from plato.tools.regressors.online_regressor import OnlineRegressor
from utils.predictors.predictor_tests import assert_online_predictor_not_broken


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


if __name__ == '__main__':
    test_gradient_descent_optimizer()
    test_adam_optimizer()
    test_adamax_optimizer()
