from abc import abstractmethod
from plato.core import add_update, create_shared_variable
from plato.interfaces.decorators import symbolic_updater
import theano.tensor as tt
import theano
import numpy as np
from plato.interfaces.helpers import get_theano_rng

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


class LangevinGradientDescent(UniformParameterOptimizer):
    """
    A simple gradient descent optimizer.  For more exotic varieties of gradient descent, use the more general
    GradientDescent class instead.
    """

    def __init__(self, eta, rng = None):
        """
        :param eta: The learning rate
        """
        self._eta = eta
        self._rng = get_theano_rng(rng)

    def _update_param(self, param, gradient):
        add_update(param, param - self._eta*gradient + 2*tt.sqrt(self._eta)*self._rng.normal(size = param.ishape))


class Adam(UniformParameterOptimizer):
    """
    The Adam optimizer.

    See paper:
    Adam: A Method for Stochastic Optimization
    Kingma D, Ba J
    http://arxiv.org/abs/1412.6980

    Adapted from
    https://gist.github.com/Newmu/acb738767acb4788bac3
    """

    def __init__(self, alpha = 1e-3, beta_1=0.1, beta_2=0.001, eps = 1e-8):
        self.alpha = alpha
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps

    def _update_param(self, param, gradient):
        # Initialize variables
        i = create_shared_variable(0.)
        m = theano.shared(param.get_value() * 0.)
        v = theano.shared(param.get_value() * 0.)

        # Recompute values
        i_t = i + 1.
        fix1 = 1. - (1. - self.beta_1)**i_t
        fix2 = 1. - (1. - self.beta_2)**i_t
        lr_t = self.alpha * (tt.sqrt(fix2) / fix1)
        m_t = (self.beta_1 * gradient) + ((1. - self.beta_1) * m)
        v_t = (self.beta_2 * tt.sqr(gradient)) + ((1. - self.beta_2) * v)
        g_t = m_t / (tt.sqrt(v_t) + self.eps)
        p_t = param - (lr_t * g_t)
        add_update(param, p_t)
        add_update(m, m_t)
        add_update(v, v_t)
        add_update(i, i_t)

#
# def Adam(cost, params, lr=0.0002, b1=0.1, b2=0.001, e=1e-8):
#     updates = []
#     grads = T.grad(cost, params)
#     i = theano.shared(floatX(0.))
#     i_t = i + 1.
#     fix1 = 1. - (1. - b1)**i_t
#     fix2 = 1. - (1. - b2)**i_t
#     lr_t = lr * (T.sqrt(fix2) / fix1)
#     for p, g in zip(params, grads):
#         m = theano.shared(p.get_value() * 0.)
#         v = theano.shared(p.get_value() * 0.)
#         m_t = (b1 * g) + ((1. - b1) * m)
#         v_t = (b2 * T.sqr(g)) + ((1. - b2) * v)
#         g_t = m_t / (T.sqrt(v_t) + e)
#         p_t = p - (lr_t * g_t)
#         updates.append((m, m_t))
#         updates.append((v, v_t))
#         updates.append((p, p_t))
#     updates.append((i, i_t))
#     return updates

class AdaMax(UniformParameterOptimizer):

    def __init__(self, alpha = 1e-3, beta_1=0.1, beta_2=0.001, eps = 1e-8):
        self._alpha = alpha
        self._beta_1 = beta_1
        self._beta_2 = beta_2
        self._eps = eps

    def _update_param(self, param, gradient):
        mom1 = theano.shared(np.zeros_like(param.get_value()))
        mom2 = theano.shared(np.zeros_like(param.get_value()))
        mom1_new = mom1 + self._beta_1 * (gradient - mom1)
        mom2_new = tt.maximum(abs(gradient) + self._eps, (1. - self._beta_2) * mom2)
        new_param = param - self._alpha * mom1_new / mom2_new
        add_update(param, new_param)
        add_update(mom1, mom1_new)
        add_update(mom2, mom2_new)


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
        add_update(param, param - (self.learning_rate / scale) * gradient)
        add_update(sum_squared_grad, new_ssg)


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
            add_update(mom, new_mom)
            direction = new_mom  # Or mom, something about Nesterov...
        else:
            direction = gradient
        add_update(param, param - self.eta*direction - self.decay*param)


class MultiplicativeGradientDescent(UniformParameterOptimizer):

    def __init__(self, factor = 0.01):
        self.factor = factor

    def _update_param(self, param, gradient):
        multiplier = tt.exp(-tt.tanh(gradient)*self.factor)
        add_update(param, param*multiplier)


# <<<<<<< HEAD
# class HMC(UniformParameterOptimizer):
#
#     def __init__(self, step_size, temperature = 1, partial_refreshment = False):
#         assert partial_refreshment, "Not set up for non-partial refreshment yet.  "
#         self.partial_refreshment = partial_refreshment
#         self.temperature = temperature
#         self.step_size = step_size
#
#     def _update_param(self, param, gradient):
#         # TODO: Actually finish this!!!
#         d_energy_d_pos = gradient * self.temperature
#         mom = create_shared_variable(np.zeros_like(param.get_value()))  # Should be random??
#         new_mom = mom - self.step_size * d_energy_d_pos
#         new_pos = param + self.step_size * new_mom
#         add_update(param, new_pos)
#         add_update(mom, new_mom)
#
# =======
# class HMC(UniformParameterOptimizer):
#
#     def __init__(self, step_size, temperature = 1, partial_refreshment = False):
#         assert partial_refreshment, "Not set up for non-partial refreshment yet.  "
#         self.partial_refreshment = partial_refreshment
#         self.temperature = temperature
#         self.step_size = step_size
#
#     def _update_param(self, param, gradient):
#         # TODO: Actually finish this!!!
#         d_energy_d_pos = gradient * self.temperature
#         mom = create_shared_variable(np.zeros_like(param.get_value()))  # Should be random??
#         new_mom = mom - self.step_size * d_energy_d_pos
#         new_pos = param + self.step_size * new_mom
#         add_update(param, new_pos)
#         add_update(mom, new_mom)
#
#
# class HMCPartial(UniformParameterOptimizer):
#
#     def __init__(self, step_size, temperature = 1, alpha = 0.99, rng = None):
#         self.temperature = temperature
#         self.step_size = step_size
#         self.alpha = alpha
#         self.rng = get_theano_rng(rng)
#
#     def _update_param(self, param, gradient):
#         # TODO: Actually finish this!!!
#         d_energy_d_pos = gradient * self.temperature
#         mom = create_shared_variable(np.zeros(param.ishape))  # Should be random??
#
#         new_mom = mom - self.step_size * d_energy_d_pos
#         new_mom = self.alpha * new_mom + tt.sqrt(1-self.alpha**2)*self.rng.normal(size=param.ishape)
#
#         new_pos = param + self.step_size * new_mom
#         add_update(param, new_pos)
#         add_update(mom, new_mom)
#
#     def metropolis_hastings_accept(self, old_energy, new_energy):

    # @staticmethod
    # def leapfrog(pos, vel, step, cost):





def get_named_optimizer(name, learning_rate, rng = None):
    """
    Convenience function for easily specifying optimizers.
    :param name: The name of the optimizer
    :param learning_rate: A scalar, representing the parameter that's most equivalent to a learning rate.
    :return: An IGradientOptimizer object.
    """
    return {
        'sgd': lambda: SimpleGradientDescent(eta = learning_rate),
        'adam': lambda: Adam(alpha=learning_rate),
        'adamax': lambda: AdaMax(alpha=learning_rate),
        'rmsprop': lambda: RMSProp(learning_rate=learning_rate),
        'adagrad': lambda: AdaGrad(learning_rate=learning_rate),
        'mulsgd': lambda: MultiplicativeGradientDescent(factor=learning_rate),
        'langevin': lambda: LangevinGradientDescent(eta = learning_rate, rng = rng),
        # 'hmc': lambda: HMC(step_size=learning_rate),
        # 'hmc-partial': lambda: HMC(step_size=learning_rate, partial_refreshment=True),
    }[name]()
