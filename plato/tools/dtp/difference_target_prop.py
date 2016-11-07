from abc import abstractmethod

import theano
import theano.tensor as tt
import numpy as np

from plato.core import symbolic_updater, symbolic_simple
from artemis.general.numpy_helpers import get_rng
from plato.interfaces.helpers import get_theano_rng, get_named_activation_function
from plato.tools.optimization.cost import mean_squared_error
from plato.tools.common.online_predictors import ISymbolicPredictor
from plato.tools.optimization.optimizers import SimpleGradientDescent
from plato.tools.common.bureaucracy import kwarg_map


__author__ = 'peter'


class ITargetPropLayer(object):

    @abstractmethod
    def predict(self, x):
        pass

    @abstractmethod
    def backward(self, x):
        pass

    @abstractmethod
    def train(self, x, target):
        pass

    @abstractmethod
    def backpropagate_target(self, x, target):
        pass


class DifferenceTargetLayer(ITargetPropLayer, ISymbolicPredictor):

    def __init__(self, w, b, w_rev, b_rev, backward_activation = 'tanh', forward_activation = 'tanh', rng = None, noise = 1,
                 optimizer_constructor = lambda: SimpleGradientDescent(0.01), cost_function = mean_squared_error, use_bias=True):

        self.noise = noise
        self.rng = get_theano_rng(rng)
        self.w = theano.shared(w, name = 'w')
        self.b = theano.shared(b, name = 'b')
        self.w_rev = theano.shared(w_rev, name = 'w_rev')
        self.b_rev = theano.shared(b_rev, name = 'b_rev')
        self.backward_activation = get_named_activation_function(backward_activation) if backward_activation is not None else None
        self.forward_activation = get_named_activation_function(forward_activation)
        self.forward_optimizer = optimizer_constructor()
        self.backward_optimizer = optimizer_constructor()
        self.cost_function = cost_function
        self.use_bias = use_bias

    @symbolic_simple
    def predict(self, x):
        return self.forward_activation(x.dot(self.w)+self.b)

    @symbolic_simple
    def backward(self, y):
        return self.backward_activation(y.dot(self.w_rev) + self.b_rev)

    @symbolic_updater
    def train(self, x, target):
        out = self.predict(x)
        self.forward_optimizer(
            cost = self.cost_function(out, target),
            parameters = [self.w, self.b] if self.use_bias else [self.w],
            constants = [target]
            )  # The "constants" (above) is really important - otherwise it can just try to change the target (which is a function of the weights too).
        noisy_x = x + self.noise*self.rng.normal(size = x.tag.test_value.shape)
        if self.backward_activation is not None:
            recon = self.backward(self.predict(noisy_x))
            self.backward_optimizer(
                cost = self.cost_function(recon, noisy_x),
                parameters = [self.w_rev, self.b_rev] if self.use_bias else [self.w_rev])

    @symbolic_simple
    def backpropagate_target(self, x, target):
        return x - self.backward(self.predict(x)) + self.backward(target)

    @classmethod
    def from_initializer(cls, n_in, n_out, w_init_mag = 0.01, rng = None, **kwargs):
        rng = get_rng(rng)
        return cls(
            w = w_init_mag*rng.randn(n_in, n_out),
            b = np.zeros(n_out),
            w_rev = w_init_mag*rng.randn(n_out, n_in),
            b_rev = np.zeros(n_in),
            rng = rng,
            **kwargs
        )


class DifferenceTargetMLP(ISymbolicPredictor):
    """
    An MLP doing Difference Target Propagation

    See:
        Dong-Hyun Lee, Saizheng Zhang, Asja Fischer, Antoine Biard1, Yoshua Bengio1
        Difference Target Propagation
        http://arxiv.org/pdf/1412.7525v3.pdf

    Notes:
    - See demo_compare_optimizers for a comparison between this and a regular MLP
    - Currently uses a eta=0.5 for the final layer.  Not sure if this is always a good
      choice - see paper.
    """

    def __init__(self, layers, output_cost_function = mean_squared_error):
        """
        (Use from_initializer constructor for more direct parametrization)
        :param layers: A list of DifferenceTargetLayers
        :param cost_function: A cost function of the form cost=cost_fcn(actual, target)
        """
        self.layers = layers
        self.output_cost_function = output_cost_function

    @symbolic_updater
    def train(self, x, target):

        xs = [x]
        for l in self.layers:
            xs.append(l.predict(xs[-1]))

        if self.output_cost_function is None:
            top_target = target
        else:
            global_loss = self.output_cost_function(xs[-1], target)
            top_target = xs[-1] - 0.5 * tt.grad(global_loss, xs[-1])
            # Note: This 0.5 thing is not always the way it should be done.
            # 0.5 is chosen because this makes it equivalent to the regular loss with MSE,
            # but it's unknown whether this is the best way to go.
        for i, l in reversed(list(enumerate(self.layers))):
            l.train(xs[i], top_target)
            if i>0:  # No need to go to initial layer
                top_target = l.backpropagate_target(xs[i], top_target)

    @symbolic_simple
    def predict(self, x):
        for l in self.layers:
            x = l.predict(x)
        return x

    @property
    def property(self):
        return

    @classmethod
    def from_initializer(cls, input_size, output_size, layer_constructor = DifferenceTargetLayer.from_initializer,
            hidden_sizes = [], hidden_activation = 'tanh', output_activation = 'softmax',
            output_cost_function = mean_squared_error, **kwargs):
        """
        :param layer_constructor: A function of the form:
            ITargetPropLayer = fcn(n_in, n_out)
        :param input_size: Size of the input
        :param hidden_sizes: List of sizes sof hidden layers
        :param output_size: Size of the output
        :return:
        """
        hidden_activations = hidden_activation if isinstance(hidden_activation, (list, tuple)) else [hidden_activation]*len(hidden_sizes)
        return cls(kwarg_map(
            lambda n_in, n_out, backward_activation, forward_activation: layer_constructor(n_in=n_in, n_out=n_out,
                backward_activation=backward_activation, forward_activation=forward_activation, **kwargs),
            n_in = [input_size] + hidden_sizes,
            n_out = hidden_sizes + [output_size],
            forward_activation = hidden_activations + [output_activation],
            backward_activation = [None] + hidden_activations,
            ), output_cost_function = output_cost_function)
