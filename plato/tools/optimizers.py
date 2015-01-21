from abc import abstractmethod
from plato.interfaces.decorators import symbolic_updater
import theano.tensor as ts
import theano
import numpy as np

__author__ = 'peter'


class IGradientOptimizer(object):

    @abstractmethod
    def __call__(self, cost, parameters):
        """
        :param cost: A scalar indicating the cost
        :return: Updates: A list of updates to whatever parameters
        """


class UniformParameterOptimizer(object):
    """
    Subclass off this if you're optimizing the same way across all parameters
    """

    def __call__(self, cost, parameters):
        return sum([self._update_param(cost, p) for p in parameters], [])

    @abstractmethod
    def _update_param(self, cost, param):
        pass


@symbolic_updater  # Temporary weirdness until things get ironed out
class SimpleGradientDescent(UniformParameterOptimizer):
    """
    A simple gradient descent optimizer.  For more exotic varieties of gradient descent, use the more general
    GradientDescent class instead.
    """

    def __init__(self, eta):
        """
        :param eta: The learning rate
        """
        self._eta = eta

    def _update_param(self, cost, param):
        return [(param, param - self._eta * theano.grad(cost, param))]


@symbolic_updater
class AdaMax(UniformParameterOptimizer):

    def __init__(self, alpha = 1e-3, beta_1=0.1, beta_2=0.001, eps = 1e-8):
        self._alpha = alpha
        self._beta_1 = beta_1
        self._beta_2 = beta_2
        self._eps = eps

    def _update_param(self, cost, param):
        mom1 = theano.shared(np.zeros_like(param.get_value()))
        mom2 = theano.shared(np.zeros_like(param.get_value()))
        gradient = theano.grad(cost, param)
        mom1_new = mom1 + self._beta_1 * (gradient - mom1)
        mom2_new = ts.maximum(abs(gradient) + self._eps, (1. - self._beta_2) * mom2)
        new_param = param - self._alpha * mom1_new / mom2_new
        updates = [(param, new_param), (mom1, mom1_new), (mom2, mom2_new)]
        return updates


class GradientDescent(object):
    """ Gradient descent, with all bells and whistles"""
