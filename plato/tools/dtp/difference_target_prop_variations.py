from artemis.general.numpy_helpers import get_rng
from plato.core import add_update
from plato.interfaces.decorators import symbolic_simple, symbolic_updater
from plato.tools.dtp.difference_target_prop import DifferenceTargetLayer, ITargetPropLayer, DifferenceTargetMLP
import numpy as np
from plato.tools.optimization.cost import mean_squared_error
import theano
from utils.bureaucracy import kwarg_map

__author__ = 'peter'



class PreActivationDifferenceTargetLayer(DifferenceTargetLayer):
    """


    We just want to see if this works (it does!  Maybe even better!)
    """

    @symbolic_simple
    def predict(self, x):
        pre_sigmoid = x.dot(self.w)+self.b
        output = self.forward_activation(pre_sigmoid)
        output.pre_sigmoid = pre_sigmoid
        return output

    def backpropagate_target(self, x, target):

        back_output_pre_sigmoid = self.predict(x).dot(self.w_rev) + self.b_rev
        back_target_pre_sigmoid = target.dot(self.w_rev) + self.b_rev
        return self.backward_activation(x.pre_sigmoid - back_output_pre_sigmoid + back_target_pre_sigmoid)


class LinearDifferenceTargetLayer(DifferenceTargetLayer):
    """
    This is an experimental modification where we switch the order of the linear/nonlinear
    operations.  That is, instead of the usual
        f(x) = activation(x.dot(w))
    We do
        f(x) = activation(x).dot(w)
    """

    @symbolic_simple
    def predict(self, x):
        return self.forward_activation(x).dot(self.w)+self.b

    @symbolic_simple
    def backward(self, y):
        return self.backward_activation(y).dot(self.w_rev) + self.b_rev


class LinearDifferenceTargetMLP(DifferenceTargetMLP):
    
    @classmethod
    def from_initializer(cls, input_size, output_size, layer_constructor = LinearDifferenceTargetLayer .from_initializer,
            hidden_sizes = [], hidden_activation = 'tanh', output_activation = 'linear',
            output_cost_function = mean_squared_error, **kwargs):
        """
        Note... we need to override the superclass from_initializer because now hidden activation is part of the backward
        """
        hidden_activations = hidden_activation if isinstance(hidden_activation, (list, tuple)) else [hidden_activation]*len(hidden_sizes)
        return cls(kwarg_map(
            lambda n_in, n_out, backward_activation, forward_activation: layer_constructor(n_in=n_in, n_out=n_out,
                backward_activation=backward_activation, forward_activation=forward_activation, **kwargs),
            n_in = [input_size] + hidden_sizes,
            n_out = hidden_sizes + [output_size],
            forward_activation = ['linear']+hidden_activations,
            backward_activation = [None]+hidden_activations[1:]+[output_activation],
            ), output_cost_function = output_cost_function)


class PerceptronLayer(ITargetPropLayer):

    def __init__(self, w, b, w_rev, b_rev, lin_dtp = True):

        self.w = theano.shared(w, name = 'w')
        self.b = theano.shared(b, name = 'b')
        self.w_rev = theano.shared(w_rev, name = 'w_rev')
        self.b_rev = theano.shared(b_rev, name = 'b_rev')
        self.lin_dtp = lin_dtp

    @symbolic_simple
    def predict(self, x):
        pre_sign = x.dot(self.w) + self.b
        output = (pre_sign > 0).astype('int32')
        output.pre_sign = pre_sign
        return output

    @symbolic_simple
    def backward(self, x):
        pre_sign = x.dot(self.w_rev) + self.b_rev
        return (pre_sign > 0).astype('int32')

    @symbolic_updater
    def train(self, x, target):

        out = self.predict(x)
        delta_w = x.T.dot(target - out)
        delta_b = (target - out).sum(axis = 0)

        recon = self.backward(out)
        delta_w_rev = out.T.dot(x - recon)
        delta_b_rev = (x - recon).sum(axis = 0)

        add_update(self.w, self.w+delta_w)
        add_update(self.w_rev, self.w_rev+delta_w_rev)
        add_update(self.b, self.b+delta_b)
        add_update(self.b_rev, self.b_rev+delta_b_rev)

    def backpropagate_target(self, x, target):

        if self.lin_dtp:
            back_output_pre_sigmoid = self.predict(x).dot(self.w_rev) + self.b_rev
            back_target_pre_sigmoid = target.dot(self.w_rev) + self.b_rev
            return (x.pre_sign - back_output_pre_sigmoid + back_target_pre_sigmoid > 0).astype('int32')
        else:
            output = self.predict(x)
            back_output = self.backward(output)
            back_target = self.backward(target)
            return x - back_output + back_target

    @classmethod
    def from_initializer(cls, n_in, n_out, initial_mag, rng=None, **kwargs):

        rng = get_rng(rng)

        return PerceptronLayer(
            w = rng.randint(low = -initial_mag, high=initial_mag+1, size = (n_in, n_out)).astype('float'),
            b = np.zeros(n_out).astype('float'),
            w_rev = rng.randint(low = -initial_mag, high=initial_mag+1, size = (n_out, n_in)).astype('float'),
            b_rev = np.zeros(n_in).astype('float'),
            **kwargs
            )
