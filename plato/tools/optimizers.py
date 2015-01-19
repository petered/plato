from abc import abstractmethod
import theano.tensor as ts
import theano

__author__ = 'peter'


class IGradientOptimizer(object):

    @abstractmethod
    def __call__(self, cost, parameters):
        """
        :param cost: A scalar indicating the cost
        :return: Updates: A list of updates to whatever parameters
        """


class SimpleGradientDescent(object):
    """
    A simple gradient descent optimizer.  For more exotic varieties of gradient descent, use the more general
    GradientDescent class instead.
    """

    def __init__(self, eta):
        """
        :param eta: The learning rate
        """
        self._eta = eta

    def __call__(self, cost, parameters):
        """
        :param cost: A scalar representing the cost.
        :param parameters: A list of shared variables.  Cost should be differentiable w.r.t. all these variables.
        :return: A list of parameter updates
        """
        gradients = [theano.grad(cost, p) for p in parameters]
        updates = [(p, p - self._eta * g) for p, g in zip(parameters, gradients)]
        return updates


class GradientDescent(object):
    """ Gradient descent, with all bells and whistles"""
