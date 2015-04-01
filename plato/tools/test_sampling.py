from plato.tools.sampling import compute_hypothetical_vs, get_p_w_given
from utils.tools.mymath import sigm

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


def dumb_compute_hypothetical_vs(x, w, alpha, (w0, w1)):
    # The obvious, wasteful way

    assert len(alpha) == 2 and len(alpha[0]) == len(alpha[1])
    n_alpha = len(alpha[0])

    obv_results = np.empty((x.shape[0], n_alpha, 2))
    for a, (ai, aj) in enumerate(zip(*alpha)):
        w_temp = w.copy()
        w_temp[ai, aj] = w0
        v_w0 = x.dot(w_temp)[:, aj]
        w_temp[ai, aj] = w1
        v_w1 = x.dot(w_temp)[:, aj]
        obv_results[:, a, 0] = v_w0
        obv_results[:, a, 1] = v_w1
    return obv_results


class Container(object):

    def __init__(self, **kwargs):
        self.__dict__.update(**kwargs)


def _get_test_data(n_samples=10, n_input_dims=20, n_output_dims=5, n_alpha=25, possible_ws = (0, 1), seed = None):
    """
    :return: Random test data
        x: An (n_samples, n_input_dims) matrix
        w: An (n_input_dims, n_output_dims) matrix with values in possible_ws
        alpha: A set of random indices (with replacement) of w
        possible_ws: The set of possible values of w.
    """
    rng = np.random.RandomState(seed)
    alpha = (rng.choice(n_input_dims, size = n_alpha), rng.choice(n_output_dims, size = n_alpha))
    x = rng.randn(n_samples, n_input_dims)
    w = rng.choice(possible_ws, size = (n_input_dims, n_output_dims))
    y = rng.choice((0, 1), size = (n_samples, n_output_dims))
    return Container(**locals())


def test_compute_hypothetical_vs():
    """

    :return:
    """
    d = _get_test_data(seed = 45)
    x, w, alpha, possible_ws = d.x, d.w, d.alpha, d.possible_ws
    # Note - assert fails with some seeds when floatX = float32
    obv_results = dumb_compute_hypothetical_vs(x, w, alpha, possible_ws)
    efficient_results = compute_hypothetical_vs.compile(fixed_args = dict(alpha=alpha, possible_ws=possible_ws))(x, w)
    assert np.allclose(obv_results, efficient_results)


def test_get_p_w_given():
    d = _get_test_data()
    p_w_alpha_w1 = get_p_w_given.compile(fixed_args = dict(alpha=d.alpha, possible_ws=d.possible_ws, boolean_ws = True))(d.x, d.w, d.y)
    assert p_w_alpha_w1.shape == (d.n_alpha,) and np.all(0 <= p_w_alpha_w1) and np.all(p_w_alpha_w1 <= 1)
    # TODO: Better test


if __name__ == '__main__':

    test_get_p_w_given()
    test_compute_hypothetical_vs()
    test_correctness_of_weight_shortcut()
