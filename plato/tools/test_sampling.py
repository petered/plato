from plato.tools.old_sampling import simple_binary_gibbs_regressor, simple_herded_binary_gibbs_regressor, \
    OldGibbsRegressor
from utils.datasets.synthetic_logistic import get_logistic_regression_data
from plato.tools.sampling import GibbsRegressor, HerdedGibbsRegressor

__author__ = 'peter'
import numpy as np


def test_correctness_of_weight_shortcut():

    # Make data with 10 samples, 20 input dims, 5 output dims
    n_samples = 10
    n_input_dims = 20
    n_output_dims = 5

    x = np.random.randn(n_samples, n_input_dims)
    w = np.random.rand(n_input_dims, n_output_dims) > 0.5

    # Say we have 4 indices in alpha indexing columns of x/rows of w.
    # We want to see what the result would be if those rows of w were
    # set to zero and one.. INDEPENDENTLY for each alpha.
    alpha = np.array([2, 5, 11, 19])

    # First- the obvious, wasteful way:
    obv_results = np.empty((len(alpha), n_samples, n_output_dims, 2))
    for i, a in enumerate(alpha):
        w_temp = w.copy()
        w_temp[a, :] = 0
        v_0 = x.dot(w_temp)
        w_temp[a, :] = 1
        v_1 = x.dot(w_temp)
        obv_results[i, :, :, 0] = v_0
        obv_results[i, :, :, 1] = v_1

    # Next, the compact and fast way
    v_current = x.dot(w)  # (n_samples, n_dim_out)
    v_0 = v_current[None] - w[alpha, None, :]*x.T[alpha, :, None]
    v_1 = v_0 + x.T[alpha, :, None]
    compact_results = np.concatenate([v_0[..., None], v_1[..., None]], axis = 3)
    assert np.allclose(obv_results, compact_results)

    # But we can do better than that
    v_current = x.dot(w)  # (n_samples, n_dim_out)
    possible_ws = np.array([0, 1])
    v_0 = v_current[None] - w[alpha, None, :]*x.T[alpha, :, None]
    super_compact_results = v_0[:, :, :, None] + possible_ws[None, None, None, :]*x.T[alpha, :, None, None]
    assert np.allclose(super_compact_results, obv_results)


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
        sampling_fcn = sampler.sample_posterior.compile()
        update_fcn = sampler.update.compile()
        for _ in xrange(2):
            out = sampling_fcn(x_tr)
            assert out.shape == (n_samples, 1)
            update_fcn(x_tr, y_tr)


if __name__ == '__main__':

    test_samplers_not_broken()
    test_correctness_of_weight_shortcut()
