from matplotlib.scale import ScaleBase
import numpy as np

__author__ = 'peter'

sigm = lambda x: 1./(1+np.exp(-x))


bernoulli = lambda k, p: (p**k)*((1-p)**(1-k))  # Maybe a not the fastest way to do it but whatevs


def sqrtspace(a, b, n_points):
    """
    :return: Distribute n_points quadratically from point a to point b, inclusive
    """
    return np.linspace(0, 1, n_points)**2*(b-a)+a


def cummean(x, axis):
    x=np.array(x)
    normalized = np.arange(1, x.shape[axis]+1).astype(float)[(slice(None), )+(None, )*(x.ndim-axis-1)]
    return np.cumsum(x, axis)/normalized
