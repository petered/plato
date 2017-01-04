from plato.core import symbolic, create_shared_variable, add_update
from plato.interfaces.decorators import symbolic_simple
from plato.interfaces.helpers import MRG_RandomStreams_ext, batchify_function
import numpy as np
import pytest

__author__ = 'peter'


@pytest.mark.skipif(True, reason="Fails on pytest but not when run directly")
def test_mrg_choice():

    n_options = 10
    n_elements = 7

    @symbolic_simple
    def random_indices():
        rng = MRG_RandomStreams_ext(seed = 4324)
        ixs = rng.choice(a=n_options, size = n_elements, replace = False)
        return ixs

    fcn = random_indices.compile()

    ixs1 = fcn()
    assert len(ixs1) == n_elements
    ixs2 = fcn()
    assert len(ixs2) == n_elements
    assert not np.array_equal(ixs1, ixs2)
    assert all(ixs1 < 10) and len(np.unique(ixs1)) == len(ixs1)
    assert all(ixs2 < 10) and len(np.unique(ixs2)) == len(ixs2)


def test_compute_in_batches():

    @symbolic
    def add_them(a, b):
        return a+b
    arr_a = np.random.randn(60, 3)  # (note that say (57, 3) would fail
    arr_b = np.random.randn(60, 3)
    f = batchify_function(add_them, batch_size=10)
    out = f.compile()(arr_a, arr_b)
    assert np.allclose(out, arr_a+arr_b)


def test_compute_in_with_state():

    @symbolic
    def add_them(a):
        i = create_shared_variable(0.)
        add_update(i, i+1)
        return a+i

    arr = np.random.randn(60, 3)
    out = batchify_function(add_them, 10).compile()(arr)
    assert all(np.allclose(out[i*10:(i+1)*10], arr[i*10:(i+1)*10]+i) for i in xrange(6))


if __name__ == '__main__':

    test_mrg_choice()
    test_compute_in_batches()
    test_compute_in_with_state()
