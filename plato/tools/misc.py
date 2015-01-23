import numpy as np


def quadspace(a, b, n_points):
    """
    :return: Distribute n_points quadratically from point a to point b, inclusive
    """
    return np.linspace(0, 1, n_points)**2*(b-a)+a
