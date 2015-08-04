from general.numpy_helpers import get_rng
from plato.core import symbolic_updater, symbolic_stateless, tdbprint
from abc import abstractmethod
from plato.interfaces.helpers import get_theano_rng, get_named_activation_function
from plato.tools.optimization.cost import mean_squared_error
from plato.tools.common.online_predictors import ISymbolicPredictor
from plato.tools.optimization.optimizers import SimpleGradientDescent
import theano
import theano.tensor as tt
import numpy as np
from utils.bureaucracy import kwarg_map

__author__ = 'peter'


class ITargetPropLayer(object):

    @abstractmethod
    def predict(self, x):
        pass

    @abstractmethod
    def backward(self, x):
        pass

    @abstractmethod
    def train(selfx, target):
        pass

    @abstractmethod
    def backpropagate_target(self, x, target):
        pass


class DifferenceTargetLayer(ITargetPropLayer, ISymbolicPredictor):

    def __init__(self, w, b, w_rev, b_rev, input_activation = 'tanh', output_activation = 'tanh', rng = None, noise = 1,
                 optimizer_constructor = lambda: SimpleGradientDescent(0.01), cost_function = mean_squared_error):

        self.noise = noise
        self.rng = get_theano_rng(rng)
        self.w = theano.shared(w, name = 'w')
        self.b = theano.shared(b, name = 'b')
        self.w_rev = theano.shared(w_rev, name = 'w_rev')
        self.b_rev = theano.shared(b_rev, name = 'b_rev')
        self.input_activation = get_named_activation_function(input_activation)
        self.hidden_activation = get_named_activation_function(output_activation)
        self.forward_optimizer = optimizer_constructor()
        self.backward_optimizer = optimizer_constructor()
        self.cost_function = cost_function

    @symbolic_stateless
    def predict(self, x):
        return self.hidden_activation(x.dot(self.w)+self.b)

    @symbolic_stateless
    def backward(self, y):
        return self.input_activation(y.dot(self.w_rev) + self.b_rev)

    @symbolic_updater
    def train(self, x, target):
        out = self.predict(x)
        forward_updates = self.forward_optimizer(
            cost = self.cost_function(out, target),
            parameters = [self.w, self.b], constants = [target]
            )  # The "constants" (above) is really important - otherwise it can just try to change the target (which is a function of the weights too).
        noisy_x = x + self.noise*self.rng.normal(size = x.tag.test_value.shape)
        recon = self.backward(self.predict(noisy_x))
        backward_updates = self.backward_optimizer(
            cost = self.cost_function(recon, noisy_x),
            parameters = [self.w_rev, self.b_rev])
        return forward_updates+backward_updates

    @symbolic_stateless
    def backpropagate_target(self, x, target):
        return x - self.backward(self.predict(x)) + self.backward(target)


class DifferenceTargetLayer(ISymbolicPredictor):

    def __init__(self, w, b, w_rev, b_rev, input_activation = 'tanh', output_activation = 'tanh', rng = None, noise = 1,
                 optimizer_constructor = lambda: SimpleGradientDescent(0.01), cost_function = mean_squared_error):

        self.noise = noise
        self.rng = get_theano_rng(rng)
        self.w = theano.shared(w, name = 'w')
        self.b = theano.shared(b, name = 'b')
        self.w_rev = theano.shared(w_rev, name = 'w_rev')
        self.b_rev = theano.shared(b_rev, name = 'b_rev')
        self.input_activation = get_named_activation_function(input_activation)
        self.hidden_activation = get_named_activation_function(output_activation)
        self.forward_optimizer = optimizer_constructor()
        self.backward_optimizer = optimizer_constructor()
        self.cost_function = cost_function

    @symbolic_stateless
    def predict(self, x):
        return self.hidden_activation(x.dot(self.w)+self.b)

    @symbolic_stateless
    def backward(self, y):
        return self.input_activation(y.dot(self.w_rev) + self.b_rev)

    @symbolic_updater
    def train(self, x, target):
        out = self.predict(x)
        forward_updates = self.forward_optimizer(
            cost = self.cost_function(out, target),
            parameters = [self.w, self.b], constants = [target]
            )  # The "constants" (above) is really important - otherwise it can just try to change the target (which is a function of the weights too).
        noisy_x = x + self.noise*self.rng.normal(size = x.tag.test_value.shape)
        recon = self.backward(self.predict(noisy_x))
        backward_updates = self.backward_optimizer(
            cost = self.cost_function(recon, noisy_x),
            parameters = [self.w_rev, self.b_rev])
        return forward_updates+backward_updates

    @symbolic_stateless
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


class ReversedDifferenceTargetLayer(DifferenceTargetLayer):
    """
    This is an experimental modification where we switch the order of the linear/nonlinear
    operations.  That is, instead of the usual
        f(x) = activation(x.dot(w))
    We do
        f(x) = activation(x).dot(w)

    We just want to see if this works.
    """

    @symbolic_stateless
    def predict(self, x):
        pre_sigmoid = x.dot(self.w)+self.b
        output = self.hidden_activation(pre_sigmoid)
        output.pre_sigmoid = pre_sigmoid
        return output

    def backpropagate_target(self, x, target):

        back_output_pre_sigmoid = self.predict(x).dot(self.w_rev) + self.b_rev
        back_target_pre_sigmoid = target.dot(self.w_rev) + self.b_rev
        return self.input_activation(x.pre_sigmoid - back_output_pre_sigmoid + back_target_pre_sigmoid)


class PerceptronLayer(object):

    def __init__(self, w, b, w_rev, b_rev):

        self.w = theano.shared(w, name = 'w')
        self.b = theano.shared(b, name = 'b')
        self.w_rev = theano.shared(w_rev, name = 'w_rev')
        self.b_rev = theano.shared(b_rev, name = 'b_rev')

    @symbolic_stateless
    def predict(self, x):
        pre_sign = x.dot(self.w) + self.b
        output = (pre_sign > 0).astype('int32')
        output.pre_sign = pre_sign
        return output

    @symbolic_stateless
    def backward(self, x):
        return (x.dot(self.w_rev) + self.b_rev > 0).astype('int32')

    @symbolic_updater
    def train(self, x, target):

        out = self.predict(x)
        delta_w = x.T.dot(target - out)
        delta_b = (target - out).sum(axis = 0)

        recon = self.backward(out)
        delta_w_rev = out.T.dot(x - recon)
        delta_b_rev = (x - recon).sum(axis = 0)

        tdbprint(tt.max(self.w), 'max-w')

        return [(self.w, self.w+delta_w), (self.b, self.b+delta_b), (self.w_rev, self.w_rev+delta_w_rev), (self.b_rev, self.b_rev+delta_b_rev)]

    def backpropagate_target(self, x, target):
        back_output_pre_sigmoid = self.predict(x).dot(self.w_rev) + self.b_rev
        back_target_pre_sigmoid = target.dot(self.w_rev) + self.b_rev
        return (x.pre_sign - back_output_pre_sigmoid + back_target_pre_sigmoid > 0).astype('int32')

    @classmethod
    def from_initializer(cls, n_in, n_out, initial_mag, rng=None):

        rng = get_rng(rng)

        return PerceptronLayer(
            w = rng.randint(low = -initial_mag, high=initial_mag+1, size = (n_in, n_out)).astype('float'),
            b = np.zeros(n_out).astype('float'),
            w_rev = rng.randint(low = -initial_mag, high=initial_mag+1, size = (n_out, n_in)).astype('float'),
            b_rev = np.zeros(n_in).astype('float'),
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

    @classmethod
    def from_initializer(cls, input_size, output_size, layer_constructor = DifferenceTargetLayer.from_initializer,
            hidden_sizes = [], input_activation = 'linear', hidden_activation = 'tanh', output_activation = 'softmax',
            **kwargs):
        """
        :param layer_constructor: A function of the form:
            ITargetPropLayer = fcn(n_in, n_out)
        :param input_size: Size of the input
        :param hidden_sizes: List of sizes of hidden layers
        :param output_size: Size of the output
        :return:
        """

        hidden_activations = [hidden_activation]*len(hidden_sizes)

        return cls(kwarg_map(
            lambda n_in, n_out, input_activation, output_activation: layer_constructor(n_in=n_in, n_out=n_out,
                input_activation=input_activation, output_activation=output_activation, **kwargs),
            n_in = [input_size] + hidden_sizes,
            n_out = hidden_sizes + [output_size],
            input_activation = [input_activation]+hidden_activations,
            output_activation = hidden_activations + [output_activation],
            ))

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

        updates = []
        for i, l in reversed(list(enumerate(self.layers))):
            local_updates = l.train(xs[i], top_target)
            updates+=local_updates
            if i>0:  # No need to go to initial layer
                top_target = l.backpropagate_target(xs[i], top_target)

        return updates

    @symbolic_stateless
    def predict(self, x):
        for l in self.layers:
            x = l.predict(x)
        return x