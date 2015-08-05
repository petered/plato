from plato.interfaces.decorators import symbolic_stateless
from theano.tensor.shared_randomstreams import RandomStreams
import numpy as np

__author__ = 'peter'


"""
Test to see if random numbers are working the way we think they are.

RandomStreams seem to be weird objects in otherwise functional theano because
they update their state implicitly.  This test just confirms that this is the case.
"""


def test_shared_random_streams():

    seed = 4

    @symbolic_stateless
    def so_random():
        rng = RandomStreams(seed = seed)
        return rng.uniform(high = 256, size = (10, ))

    random_ints = so_random.compile()
    a = random_ints()
    b = random_ints()

    random_ints = so_random.compile()
    aa = random_ints()
    bb = random_ints()

    assert np.array_equal(a, aa)
    assert np.array_equal(b, bb)
    assert not np.array_equal(a, b)


if __name__ == '__main__':

    test_shared_random_streams()
