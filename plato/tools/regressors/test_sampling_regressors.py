from pytest import raises

from plato.tools.optimization.old_sampling import simple_binary_gibbs_regressor, simple_herded_binary_gibbs_regressor, \
    OldGibbsRegressor
from plato.tools.regressors.simple_sampling_regressors import GibbsRegressor, HerdedGibbsRegressor
from artemis.ml.datasets.synthetic_logistic import get_logistic_regression_data
from artemis.ml.predictors.predictor_tests import assert_online_predictor_not_broken


__author__ = 'peter'


def test_samplers_not_broken():
    """
    Just test that samplers don't break.  This doesn't assert that they actually work as
    they're meant to.  It's like testing that the car starts - it still may be a bad car,
    but at least it starts.
    """

    n_samples = 30
    n_dims = 20

    x_tr, y_tr, _, _, _ = get_logistic_regression_data(n_dims = n_dims,
        n_training=n_samples, n_test=15, noise_factor = 0.1)

    samplers = {
        'gibbs': GibbsRegressor(n_dim_in=n_dims, n_dim_out=1, n_alpha=3, possible_ws = [-1, 0, 1]),
        'herded-gibbs': HerdedGibbsRegressor(n_dim_in=n_dims, n_dim_out=1, n_alpha=3, possible_ws = [-1, 0, 1]),
        'simple-gibbs': simple_binary_gibbs_regressor(n_dim_in=n_dims, n_dim_out=1),
        'simple-herded-gibbs': simple_herded_binary_gibbs_regressor(n_dim_in=n_dims, n_dim_out=1),
        'old-gibbs': OldGibbsRegressor(n_dim_in=n_dims, n_dim_out=1)
        }

    for name, sampler in samplers.iteritems():
        print 'Running Test for Sampler %s' % name
        sampling_fcn = sampler.predict.compile(add_test_values = True)
        update_fcn = sampler.train.compile(add_test_values = True)
        for _ in xrange(2):
            out = sampling_fcn(x_tr)
            assert out.shape == (n_samples, 1)
            update_fcn(x_tr, y_tr)


def test_gibbs_logistic_regressor():

    assert_online_predictor_not_broken(
        predictor_constructor = lambda n_dim_in, n_dim_out:
            GibbsRegressor(n_dim_in = n_dim_in, n_dim_out = n_dim_out,
                n_alpha = 1,
                possible_ws= (-1, 1),
                seed = 2143
                ).compile(add_test_values = True),
        n_extra_tests = 8,
        n_epochs=20
        )


def test_herded_logistic_regressor():

    assert_online_predictor_not_broken(
        predictor_constructor = lambda n_dim_in, n_dim_out:
            HerdedGibbsRegressor(n_dim_in = n_dim_in, n_dim_out = n_dim_out,
                n_alpha = 1,
                possible_ws= (-1, 1),
                ).compile(add_test_values = True),
        n_epochs=20
        )


def test_gibbs_logistic_regressor_full_update():
    """
    This test just demonstrates that you can't just go and update all the weights at once -
    it won't work.
    """

    with raises(AssertionError):
        assert_online_predictor_not_broken(
            predictor_constructor = lambda n_dim_in, n_dim_out:
                GibbsRegressor(n_dim_in = n_dim_in, n_dim_out = n_dim_out,
                    n_alpha = n_dim_in,  # All weights updated in one go.
                    possible_ws= (-1, 1),
                    seed = 2143
                    ).compile(add_test_values = True),
            n_epochs=80
            )


if __name__ == '__main__':
    test_gibbs_logistic_regressor_full_update()
    test_herded_logistic_regressor()
    test_gibbs_logistic_regressor()
    test_samplers_not_broken()
