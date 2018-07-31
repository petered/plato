from abc import abstractmethod

from theano.ifelse import ifelse

from plato.core import add_update, create_shared_variable, StateCatcher, tdbprint, CaptureUpdates, symbolic_stateless
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

    @abstractmethod
    def get_updates(self, cost, parameters, constants = ()):
        """
        :param Scalar cost:
        :param Sequence[Variable] parameters:
        :param Sequence[Variable] constants:
        :return Sequence[Tuple[Tensor, Tensor]]: Pairs of (variable, new_variable)
        """
        pass

    @abstractmethod
    def get_updates_from_gradients(self, parameters, gradients):
        """
        :param Sequence[Tensor] parameters:
        :param Sequence[Tensor] gradients:
        :return Sequence[Tuple[Tensor, Tensor]]:
        """

    @abstractmethod
    def update_parameters(self, cost, parameters, constants=()):
        pass

    @abstractmethod
    def update_from_gradients(self, parameters, gradients, clip=None):
        pass


@symbolic_updater
class UniformParameterOptimizer(IGradientOptimizer):
    """
    Subclass off this if you're optimizing the same way across all parameters
    """

    def __call__(self, cost, parameters, constants = []):
        """
        Compute gradient-based parameter updates, and apply them.
        Deprecated - use the more clear "update_parameters"
        """
        self.update_parameters(cost=cost, parameters=parameters, constants=constants)

    def get_updates(self, cost, parameters, constants = [], clip=None):
        """
        Get the gradient-based parameter updates, but do not apply them.
        return: A list of (shared_var, new_val) pairs representing the updates.
        """
        gradients = theano.grad(cost, parameters, consider_constant = constants)
        return self.get_updates_from_gradients(parameters=parameters, gradients=gradients, clip=clip)

    def update_parameters(self, cost, parameters, constants = []):
        """
        Compute gradient-based parameter updates, and apply them.
        """
        gradients = theano.grad(cost, parameters, consider_constant = constants)  # Can be faster than [theano.grad(p) for p in parameters]
        self.update_from_gradients(parameters, gradients)

    @symbolic_updater
    def update_from_gradients(self, parameters, gradients, clip = None):
        """
        A secondary entry point (if for whatever reason you want to get the gradients yourself (e.g. if it's some kind
        of pseudo-gradient)) use this.
        :param parameters: A list of shared variables
        :param gradients: A list of corresponding gradients
        :param clip: Optionally, a 2-tuple indicating the range in which to clip parameters, (or
        """
        updates = self.get_updates_from_gradients(parameters=parameters, gradients=gradients, clip=clip)
        for p, v in updates:
            add_update(p, v)

    @symbolic_stateless
    def get_updates_from_gradients(self, parameters, gradients, clip=None):
        """
        :param Sequence[Variable] parameters: The list of symbolic parameters
        :param Sequence[Variable] gradients: The list of gradients
        :param Optional[Union[float, Tuple[float,float]] clip: The clipping parameter
        :return Sequence[Tuple[Variable, Variable]]: The list of updates (the first len(parameters) of which are ordered parameter updates - the rest are for optimizer params).
        """
        if clip is not None and not isinstance(clip, (list, tuple)):
            clip = (-clip, clip)
        assert len(parameters)==len(gradients), 'Lenght of parameter vector must match length of gradients.'
        parameter_updates_list = []
        optimizer_updates_list = []
        for p, g in zip(parameters, gradients):
            updates = self._get_updates_for_param(p, g)
            param_update = updates[0] if clip is None else (updates[0][0], tt.clip(updates[0][1], *clip))
            parameter_updates_list.append(param_update)
            optimizer_updates_list += updates[1:]
        all_updates = parameter_updates_list + optimizer_updates_list
        return all_updates

    @abstractmethod
    def _get_updates_for_param(self, param, gradient):
        """
        A stateless method
        :param Variable param: The parameter
        :param Variable gradient: The gradient of this parameter
        :return Sequence[Tuple[Variable, Variable]]: The updates - the first of which is the parameter updates (others may update optimizer state)
        """


class GradientStepUpdater(UniformParameterOptimizer):
    """
    Just subtract the gradient to the parameter.  This is mainly useful in some situations the step size doesn't matter
    (because for instance, the function is invariant to the scale of the weights)
    """
    def _get_updates_for_param(self, param, gradient):
        return [(param, param-gradient)]
        # add_update(param, param - gradient)


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

    def _get_updates_for_param(self, param, gradient):
        return [(param, param - self._eta * gradient)]
        # add_update(param, param - self._eta * gradient)


def create_optimizer_param_like(param, name=None):
    """
    :param TensorVariable like: A variable which it is "like"
    :return Tuple[TensorSharedVariable, Scalar]: The variable and a scalar boolean tensor that can be used in an ifelse to check if its been initialized.
    """
    opt_param = theano.shared(np.zeros([0]*param.ndim, dtype=param.dtype), name=name)
    initialized = opt_param.size>0
    return opt_param, initialized


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

    def _get_updates_for_param(self, param, gradient):
        # add_update(param, param - self._eta*gradient + 2*tt.sqrt(self._eta)*self._rng.normal(size = param.ishape))
        return[(param, param - self._eta*gradient + 2*tt.sqrt(self._eta)*self._rng.normal(size = param.ishape))]


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

    def _get_updates_for_param(self, param, gradient):
        # Initialize variables
        i = create_shared_variable(0.)
        # m = theano.shared(param.get_value() * 0.)
        # v = theano.shared(param.get_value() * 0.)

        m, initialized = create_optimizer_param_like(param)
        v, _ = create_optimizer_param_like(param)
        # v = theano.shared(param.ndim * 0.)

        # Recompute values
        i_t = i + 1.
        fix1 = 1. - (1. - self.beta_1)**i_t
        fix2 = 1. - (1. - self.beta_2)**i_t
        lr_t = self.alpha * (tt.sqrt(fix2) / fix1)
        m_t = ifelse(initialized, self.beta_1 * gradient + (1. - self.beta_1) * m, self.beta_1 * gradient)
        v_t = ifelse(initialized, self.beta_2 * tt.sqr(gradient) + (1. - self.beta_2) * v, self.beta_2 * tt.sqr(gradient))
        g_t = m_t / (tt.sqrt(v_t) + self.eps)
        p_t = param - (lr_t * g_t)
        return [(param, p_t), (m, m_t), (v, v_t), (i, i_t)]

        # add_update(param, p_t)
        # add_update(m, m_t)
        # add_update(v, v_t)
        # add_update(i, i_t)


class AdaMax(UniformParameterOptimizer):

    def __init__(self, alpha = 1e-3, beta_1=0.1, beta_2=0.001, eps = 1e-8):
        self._alpha = alpha
        self._beta_1 = beta_1
        self._beta_2 = beta_2
        self._eps = eps

    def _get_updates_for_param(self, param, gradient):

        mom1, initialized = create_optimizer_param_like(param)
        mom2, _ = create_optimizer_param_like(param)

        # mom1 = theano.shared(np.zeros_like(param.get_value()))
        # mom2 = theano.shared(np.zeros_like(param.get_value()))
        mom1_new = ifelse(initialized, mom1 + self._beta_1 * (gradient - mom1), self._beta_1*gradient)
        mom2_new = ifelse(initialized, tt.maximum(abs(gradient) + self._eps, (1. - self._beta_2) * mom2), abs(gradient) + self._eps)
        new_param = param - self._alpha * mom1_new / mom2_new
        return [(param, new_param), (mom1, mom1_new), (mom2, mom2_new)]

        # add_update(param, new_param)
        # add_update(mom1, mom1_new)
        # add_update(mom2, mom2_new)


class RMSProp(UniformParameterOptimizer):

    def __init__(self, learning_rate = 0.1, decay = 0.9, max_scaling = 1e5):
        self.decay = decay
        self.epsilon = 1./max_scaling
        self.learning_rate = learning_rate

    def _get_updates_for_param(self, param, gradient):
        # mean_squared_grad = theano.shared(np.zeros_like(param.get_value()))
        mean_squared_grad, initialized = create_optimizer_param_like(param)

        new_mean_squared_grad = ifelse(initialized, self.decay * mean_squared_grad + (1-self.decay) * gradient**2, (1-self.decay) * gradient**2)
        delta_p = - self.learning_rate * gradient / tt.maximum(tt.sqrt(new_mean_squared_grad), self.epsilon)

        return [(param, param + delta_p), (mean_squared_grad, new_mean_squared_grad)]
        # add_update(param, param + delta_p)
        # add_update(mean_squared_grad, new_mean_squared_grad)


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

    def _get_updates_for_param(self, param, gradient):
        # sum_squared_grad = theano.shared(param.get_value()*0)

        sum_squared_grad, initialized = create_optimizer_param_like(param)

        new_ssg = ifelse(initialized, (1-self.decay_rate)*sum_squared_grad + gradient**2, gradient**2)
        scale = tt.maximum(self.eps, tt.sqrt(new_ssg))
        return [(param, param - (self.learning_rate / scale) * gradient), (sum_squared_grad, new_ssg)]
        # add_update(param, param - (self.learning_rate / scale) * gradient)
        # add_update(sum_squared_grad, new_ssg)


class GradientDescent(UniformParameterOptimizer):
    """ Gradient descent, with all bells and whistles"""

    def __init__(self, eta, momentum = 0, decay = 0):
        """
        :param eta: The learning rate
        """
        self.eta = eta
        self.momentum = momentum
        self.decay = decay

    def _get_updates_for_param(self, param, gradient):

        updates = []

        if self.momentum != 0:
            mom, initialized = create_optimizer_param_like(param)
            # mom = theano.shared(np.zeros_like(param.get_value()))
            new_mom = ifelse(initialized, self.momentum * mom + gradient, gradient)
            # add_update(mom, new_mom)
            updates.append((mom, new_mom))
            direction = new_mom  # Or mom, something about Nesterov...
        else:
            direction = gradient

        updates.insert(0, (param, param - self.eta*direction - self.decay*param))
        return updates
        # add_update(param, param - self.eta*direction - self.decay*param)


class MultiplicativeGradientDescent(UniformParameterOptimizer):

    def __init__(self, factor = 0.01):
        self.factor = factor

    def _get_updates_for_param(self, param, gradient):
        multiplier = tt.exp(-tt.tanh(gradient)*self.factor)
        return [(param, param*multiplier)]
        # add_update(param, param*multiplier)


class PIDOptimizer(UniformParameterOptimizer):
    """ Gradient descent, with all bells and whistles"""

    def __init__(self, kp=0.1, ki=0, kd=0):
        """
        :param eta: The learning rate
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd

    def _get_updates_for_param(self, param, gradient):

        updates = []
        new_param = param
        if self.kp != 0:
            new_param -= self.kp * gradient
        if self.ki != 0:
            grad_integral = create_shared_variable(np.zeros_like(param.get_value()))
            new_gradient_integral = grad_integral + grad_integral
            # add_update(grad_integral, new_gradient_integral)
            updates.append((grad_integral, new_gradient_integral))
            new_param -= self.ki * new_gradient_integral
        if self.kd != 0:
            grad_last = create_shared_variable(np.zeros_like(param.get_value()))
            # add_update(grad_last, gradient)
            updates.append((grad_last, gradient))
            new_param -= self.kd * (gradient - grad_last)
        # add_update(param, new_param)
        updates.insert(0, (param, new_param))
        return updates


def get_named_optimizer(name, learning_rate, rng = None, **kwargs):
    """
    Convenience function for easily specifying optimizers.
    :param name: The name of the optimizer
    :param learning_rate: A scalar, representing the parameter that's most equivalent to a learning rate.
    :return IGradientOptimizer: The optimizer object.
    """
    return {
        'sgd': lambda: GradientDescent(eta = learning_rate, **kwargs),
        'adam': lambda: Adam(alpha=learning_rate, **kwargs),
        'adamax': lambda: AdaMax(alpha=learning_rate, **kwargs),
        'rmsprop': lambda: RMSProp(learning_rate=learning_rate, **kwargs),
        'adagrad': lambda: AdaGrad(learning_rate=learning_rate, **kwargs),
        'mulsgd': lambda: MultiplicativeGradientDescent(factor=learning_rate, **kwargs),
        'langevin': lambda: LangevinGradientDescent(eta = learning_rate, rng = rng, **kwargs),
    }[name]()
