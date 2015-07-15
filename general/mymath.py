from general.should_be_builtins import memoize
import numpy as np
from scipy.stats import norm
__author__ = 'peter'

# Note - this module used to be called math, but it somehow results in a numpy import error
# due to some kind of name conflict with another module called math.

sigm = lambda x: 1/(1+np.exp(-x))


def cummean(x, axis = None):
    """
    Cumulative mean along axis
    :param x: An array
    :param axis: The axis
    :return: An array of the same shape
    """
    if axis is None:
        assert isinstance(x, list) or x.ndim == 1, 'You must specify axis for a multi-dimensional array'
        axis = 0
    elif axis < 0:
        axis = x.ndim+axis
    x = np.array(x)
    normalizer = np.arange(1, x.shape[axis]+1).astype(float)[(slice(None), )+(None, )*(x.ndim-axis-1)]
    return np.cumsum(x, axis)/normalizer


def cumvar(x, axis = None, sample = True):
    """
    :return: Cumulative variance along axis
    """
    if axis is None:
        assert isinstance(x, list) or x.ndim == 1, 'You must specify axis for a multi-dimensional array'
        axis = 0
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    ex_2 = cummean(x, axis=axis)**2
    e_x2 = cummean(x**2, axis=axis)
    var = e_x2-ex_2
    if sample and x.shape[axis] > 1:
        var *= x.shape[axis]/float(x.shape[axis]-1)
    return var


@memoize
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


def expected_sigm_of_norm(mean, std, method = 'probit'):
    """
    Approximate the expected value of the sigmoid of a normal distribution.

    Thanks go to this guy:
    http://math.stackexchange.com/questions/207861/expected-value-of-applying-the-sigmoid-function-to-a-normal-distribution

    :param mean: Mean of the normal distribution
    :param std: Standard Deviation of the normal distribution
    :return: An approximation to Expectation(sigm(N(mu, sigma**2)))
    """
    if method == 'maclauren-2':
        eu = np.exp(-mean)
        approx_exp = 1/(eu+1) + 0.5*(eu-1)*eu/((eu+1)**3) * std**2
        return np.minimum(np.maximum(approx_exp, 0), 1)

    elif method == 'maclauren-3':
        eu = np.exp(-mean)
        approx_exp = 1/(eu+1) + \
            0.5*(eu-1)*eu/((eu+1)**3) * std**2 + \
            (eu**3-11*eu**2+57*eu-1)/((8*(eu+1))**5) * std**4
        return np.minimum(np.maximum(approx_exp, 0), 1)
    elif method == 'probit':
        return norm.cdf(mean/np.sqrt(2.892 + std**2))
    else:
        raise Exception('Method "%s" not known' % method)


l1_error = lambda x1, x2: np.mean(np.abs(x1-x2), axis = 1)


def normalize(x, axis=None, degree = 2):
    return x/(np.sum(x**degree, axis = axis, keepdims=True))**(1./degree)
