from plato.tools.old_sampling import simple_binary_gibbs_regressor, simple_herded_binary_gibbs_regressor, \
    OldGibbsRegressor
from plato.tools.simple_sampling_regressors import GibbsRegressor, HerdedGibbsRegressor
from utils.datasets.synthetic_logistic import get_logistic_regression_data

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
        sampling_fcn = sampler.predict.compile()
        update_fcn = sampler.train.compile()
        for _ in xrange(2):
            out = sampling_fcn(x_tr)
            assert out.shape == (n_samples, 1)
            update_fcn(x_tr, y_tr)


if __name__ == '__main__':

    test_samplers_not_broken()
