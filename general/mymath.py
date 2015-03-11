import numpy as np

__author__ = 'peter'

# Note - this module used to be called math, but it somehow results in a numpy import error
# due to some kind of name conflict with another module called math.

sigm = lambda x: 1/(1+np.exp(-x))


def cummean(x, axis):
    x=np.array(x)
    normalized = np.arange(1, x.shape[axis]+1).astype(float)[(slice(None), )+(None, )*(x.ndim-axis-1)]
    return np.cumsum(x, axis)/normalized


def binary_permutations(n_bits):
    """
    Given some number of bits, return a shape (2**n_bits, n_bits) boolean array containing every permoutation
    of those bits as a row.
    :param n_bits: An integer number of bits
    :return: A shape (2**n_bits, n_bits) boolean array containing every permoutation
        of those bits as a row.
    """
    return np.right_shift(np.arange(2**n_bits)[:, None], np.arange(n_bits-1, -1, -1)[None, :]) & 1


def softmax(x, axis = None):
    """
    The softmax function takes an ndarray, and returns an ndarray of the same size,
    with the softmax function applied along the given axis.  It should always be the
    case that np.allclose(np.sum(softmax(x, axis), axis)==1)
    """
    if axis is None:
        assert x.ndim==1, "You need to specify the axis for softmax if your data is more thn 1-D"
        axis = 0
    x = x - np.max(x, axis=axis, keepdims=True)  # For numerical stability - has no effect mathematically
    expx = np.exp(x)
    return expx/np.sum(expx, axis=axis, keepdims=True)
