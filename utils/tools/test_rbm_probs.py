from utils.tools.rbm_probs import estimate_log_z, compute_exact_log_z

__author__ = 'peter'
import numpy as np


def test_partition_estimate():

    seed = 1234
    mag = 1
    n_visible = 200
    n_hidden = 10

    rng = np.random.RandomState(seed)

    b_v = mag * rng.randn(n_visible)
    b_h = mag * rng.randn(n_hidden)
    w = mag * rng.randn(n_visible, n_hidden)

    exact_partition = compute_exact_log_z(w=w, b_v=b_v, b_h = b_h)
    approx_partition, (upper, lower) = estimate_log_z(w=w, b_h=b_h, b_v=b_v, annealing_ratios=np.linspace(0, 1, 1000), rng = None)
    assert np.allclose(exact_partition, approx_partition, atol = 0, rtol = 0.1)  # Very approximate!


if __name__ == '__main__':

    test_partition_estimate()
