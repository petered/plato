from plato.tools.optimization.optimizers import GradientDescent
from plato.tools.regressors.demo_mnist_regression import demo_mnist_online_regression
from plato.tools.regressors.online_regressor import OnlineRegressor
from utils.predictors.predictor_tests import assert_online_predictor_not_broken

__author__ = 'peter'


def test_online_regressors():

    # Multinomial Regression
    assert_online_predictor_not_broken(
        predictor_constructor = lambda n_dim_in, n_dim_out:
            OnlineRegressor(
                input_size = n_dim_in,
                output_size=n_dim_out,
                optimizer=GradientDescent(eta = 0.01),
                regressor_type = 'multinomial'
                ).compile(),
        categorical_target=True,
        n_epochs=20
        )

    # Logistic Regression
    assert_online_predictor_not_broken(
        predictor_constructor = lambda n_dim_in, n_dim_out:
            OnlineRegressor(
                input_size = n_dim_in,
                output_size=n_dim_out,
                optimizer=GradientDescent(eta = 0.01),
                regressor_type = 'logistic'
                ).compile(),
        categorical_target=False,
        n_epochs=20
        )

    # Linear Regression
    assert_online_predictor_not_broken(
        predictor_constructor = lambda n_dim_in, n_dim_out:
            OnlineRegressor(
                input_size = n_dim_in,
                output_size=n_dim_out,
                optimizer=GradientDescent(eta = 0.01),
                regressor_type = 'linear'
                ).compile(),
        categorical_target=False,
        n_epochs=20
        )


def test_demo_mnist_regression():
    demo_mnist_online_regression()