from plato.interfaces.decorators import symbolic_stateless
from plato.interfaces.helpers import MRG_RandomStreams_ext
import numpy as np
import pytest

__author__ = 'peter'


@pytest.mark.skipif(True, reason="Fails on pytest but not when run directly")
def test_mrg_choice():

    n_options = 10
    n_elements = 7

    @symbolic_stateless
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


if __name__ == '__main__':

    test_mrg_choice()
