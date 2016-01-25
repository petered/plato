from abc import abstractmethod
from plato.core import add_update
from plato.interfaces.decorators import symbolic_updater
import theano.tensor as tt
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

@symbolic_updater
class UniformParameterOptimizer(IGradientOptimizer):
    """
    Subclass off this if you're optimizing the same way across all parameters
    """

    def __call__(self, cost, parameters, constants = []):
        gradients = theano.grad(cost, parameters, consider_constant = constants)  # Can be faster than [theano.grad(p) for p in parameters]
        self.update_from_gradients(parameters, gradients)

    @symbolic_updater
    def update_from_gradients(self, parameters, gradients):
        """
        A secondary entry point (if for whatever reason you want to get the gradients yourself (e.g. if it's some kind
        of pseudo-gradient)) use this.
        """
        assert len(parameters)==len(gradients), 'Lenght of parameter vector must match length of gradients.'
        for p, g in zip(parameters, gradients):
            self._update_param(p, g)

    @abstractmethod
    def _update_param(self, param, gradient):
        pass


class GradientStepUpdater(UniformParameterOptimizer):
    """
    Just subtract the gradient to the parameter.  This is mainly useful in some situations the step size doesn't matter
    (because for instance, the function is invariant to the scale of the weights)
    """
    def _update_param(self, param, gradient):
        add_update(param, param - gradient)
        # return [(param, param - gradient)]


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

    def _update_param(self, param, gradient):
        add_update(param, param - self._eta * gradient)


class AdaMax(UniformParameterOptimizer):

    def __init__(self, alpha = 1e-3, beta_1=0.1, beta_2=0.001, eps = 1e-8):
        self._alpha = alpha
        self._beta_1 = beta_1
        self._beta_2 = beta_2
        self._eps = eps

    def _update_param(self, param, gradient):
        mom1 = theano.shared(np.zeros_like(param.get_value()))
        mom2 = theano.shared(np.zeros_like(param.get_value()))
        # gradient = theano.grad(cost, param, consider_constant = constants)
        mom1_new = mom1 + self._beta_1 * (gradient - mom1)
        mom2_new = tt.maximum(abs(gradient) + self._eps, (1. - self._beta_2) * mom2)
        new_param = param - self._alpha * mom1_new / mom2_new
        add_update(param, new_param)
        add_update(mom1, mom1_new)
        add_update(mom2, mom2_new)
        # return updates


class RMSProp(UniformParameterOptimizer):

    def __init__(self, learning_rate = 0.1, decay = 0.9, max_scaling = 1e5):
        self.decay = decay
        self.epsilon = 1./max_scaling
        self.learning_rate = learning_rate

    def _update_param(self, param, gradient):
        mean_squared_grad = theano.shared(np.zeros_like(param.get_value()))
        new_mean_squared_grad = self.decay * mean_squared_grad + (1-self.decay) * gradient**2
        delta_p = - self.learning_rate * gradient / tt.maximum(tt.sqrt(new_mean_squared_grad), self.epsilon)
        add_update(param, param + delta_p)
        add_update(mean_squared_grad, new_mean_squared_grad)
        # return [(param, param + delta_p), (mean_squared_grad, new_mean_squared_grad)]


class AdaGrad(UniformParameterOptimizer):
    """
    Adaptive Learning Rate Method
    "Adaptive subgradient methods for online learning and
    stochastic optimization", Duchi J, Hazan E, Singer Y.

    Adapted from pylearn2 code:
    https://github.com/lisa-lab/pylearn2/blob/master/pylearn2/training_algorithms/learning_rule.py
    """

    def __init__(self, learning_rate = 0.01, decay_rate = 0, max_scaling = 1e5):
        self.eps = 1./max_scaling
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate

    def _update_param(self, param, gradient):
        sum_squared_grad = theano.shared(param.get_value()*0)
        new_ssg = (1-self.decay_rate)*sum_squared_grad + gradient**2
        scale = tt.maximum(self.eps, tt.sqrt(new_ssg))
        new_param = param - (self.learning_rate / scale) * gradient
        add_update(param, new_param)
        add_update(sum_squared_grad, new_ssg)
        # return [(param, new_param), (sum_squared_grad, new_ssg)]


class GradientDescent(UniformParameterOptimizer):
    """ Gradient descent, with all bells and whistles"""

    def __init__(self, eta, momentum = 0, decay = 0):
        """
        :param eta: The learning rate
        """
        self.eta = eta
        self.momentum = momentum
        self.decay = decay

    def _update_param(self, param, gradient):

        if self.momentum != 0:
            mom = theano.shared(np.zeros_like(param.get_value()))
            new_mom = self.momentum * mom + gradient
            # momentum_updates = [(mom, new_mom)]
            add_update(mom, new_mom)
            direction = new_mom  # Or mom, something about Nesterov...
        else:
            direction = gradient
            # momentum_updates = []

        add_update(param, param - self.eta*direction - self.decay*param)
        # return [(param, param - self.eta*direction - self.decay*param)] + momentum_updates


class MultiplicativeGradientDescent(UniformParameterOptimizer):

    def __init__(self, factor = 0.01):
        self.factor = factor

    def _update_param(self, param, gradient):
        multiplier = tt.exp(-tt.tanh(gradient)*self.factor)
        add_update(param, param*multiplier)
        # return [(param, param*multiplier)]


def get_named_optimizer(name, learning_rate):
    """
    Convenience function for easily specifying optimizers.
    :param name: The name of the optimizer
    :param learning_rate: A scalar, representing the parameter that's most equivalent to a learning rate.
    :return: An IGradientOptimizer object.
    """
    return {
        'sgd': SimpleGradientDescent(eta = learning_rate),
        'adamax': AdaMax(alpha=learning_rate),
        'rmsprop': RMSProp(learning_rate=learning_rate),
        'adagrad': AdaGrad(learning_rate=learning_rate),
        'mulsgd': MultiplicativeGradientDescent(factor=learning_rate)
    }[name]
