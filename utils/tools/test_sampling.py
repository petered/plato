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


if __name__ == '__main__':

    test_correctness_of_weight_shortcut()