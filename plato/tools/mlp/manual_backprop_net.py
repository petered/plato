from abc import abstractmethod
from collections import OrderedDict

import numpy as np

from artemis.general.nested_structures import get_leaf_values, NestedType
from artemis.general.should_be_builtins import izip_equal
from plato.core import create_constant, symbolic
from plato.interfaces.helpers import batchify_function, get_named_activation_function, get_parameters_or_not
from plato.interfaces.interfaces import IParameterized
from plato.tools.common.online_predictors import ISymbolicPredictor
from plato.tools.mlp.mlp import FullyConnectedTransform
from plato.tools.optimization.cost import get_named_cost_function
from plato.tools.optimization.optimizers import IGradientOptimizer
from theano import tensor as tt


class ManualBackpropNet(ISymbolicPredictor):
    """
    A sequential (chain) network where you can insert layers that do backprop manually.
    """
    def __init__(self, layers, optimizer, loss, prediction_minibatch_size=None, pass_loss = True, params_to_train = None):
        """
        :param layrs:
        :param optimizer:
        :param loss:
        """
        if isinstance(layers, (OrderedDict, list, tuple)): # Backwards compatibility baby!
            self.model = ChainNetwork(layers)
        else:
            self.model = layers
        self.optimizer = optimizer
        self.pass_loss = pass_loss
        self.loss = get_named_cost_function(loss) if isinstance(loss, basestring) else loss
        self.prediction_minibatch_size = prediction_minibatch_size
        self.params_to_train = params_to_train

    @symbolic
    def predict(self, x):
        if self.prediction_minibatch_size is None:
            return self._predict_in_single_pass(x)
        else:
            return batchify_function(self._predict_in_single_pass, batch_size=self.prediction_minibatch_size)(x)

    def _predict_in_single_pass(self, x):
        out, _ = self.model.forward_pass_and_state(x)
        return out

    @symbolic
    def train(self, x, y):
        out, state = forward_pass_and_state(self.model, x)
        loss = self.loss(out, y)
        if self.pass_loss:
            grad = None
        else:
            grad = tt.grad(loss, wrt=out)
            loss = None
        _, param_grad_pairs = backward_pass(self.model, state=state, grad=grad, loss=loss)

        if self.params_to_train is not None:
            params_in_net = set(p for p, g in param_grad_pairs)
            assert params_in_net.issuperset(self.params_to_train), 'You listed parameters to train {} which were not in the model'.format(set(self.params_to_train).difference(params_in_net))
            param_grad_pairs = [(p, g) for p, g in param_grad_pairs if p in self.params_to_train]

        if isinstance(self.optimizer, IGradientOptimizer):
            all_params, all_param_grads = zip(*[(p, g) for p, g in param_grad_pairs]) if len(param_grad_pairs)>0 else ([], [])
            self.optimizer.update_from_gradients(parameters=all_params, gradients=all_param_grads)
        elif isinstance(self.optimizer, (list, tuple)):
            for optimizer, layer_pairs in izip_equal(self.optimizer, param_grad_pairs):
                params, grads = zip(*layer_pairs)
                optimizer.update_from_gradients(parameters=params, gradients=grads)

    @property
    def parameters(self):
        return self.model.parameters


@symbolic
class IManualBackpropLayer(IParameterized):

    def forward_pass(self, x):
        """
        :param x: A real (n_samples, n_dims_in) input
        :return: A real (n_samples, n_dims_in) output
        """
        out, _ = self.forward_pass_and_state(x)
        return out

    def __call__(self, *args):
        return self.forward_pass(*args)

    @abstractmethod
    def forward_pass_and_state(self, x):
        """
        :param x:
        :return: out, state
            Where:
                out is the output of the layer
                state is a list of state-variables to be passed into the backward pass.
                Importantly, they must be in order (so that the last element of state is the one used to compute the gradient)
        """
        raise NotImplementedError()

    @abstractmethod
    def backward_pass(self, state, grad, loss):
        """
        :param state: The list of state variables you returned in forward_pass_and_state
        :param grad: The incoming gradient
        :return: The outgoing gradient
        """
        raise NotImplementedError()


def forward_pass_and_state(layer, x):
    if isinstance(layer, IManualBackpropLayer):
        out, layer_state = layer.forward_pass_and_state(x)
    else:
        out = layer(x)
        layer_state = (x, out)
    return out, layer_state


def backward_pass(layer, state, grad, loss):
    if isinstance(layer, IManualBackpropLayer):
        grad_inputs, param_grad_pairs = layer.backward_pass(state=state, grad=grad, loss= loss)
    else:
        inputs, y = state
        params = list(get_parameters_or_not(layer))
        grad_inputs = tt.grad(cost=loss, wrt=inputs, known_grads={y: grad} if grad is not None else None)
        grad_params = tt.grad(cost=loss, wrt=params, known_grads={y: grad} if grad is not None else None)
        param_grad_pairs = [(p, g) for p, g in izip_equal(params, grad_params)]
    return grad_inputs, param_grad_pairs


class ChainNetwork(IManualBackpropLayer):

    def __init__(self, layers):
        if isinstance(layers, OrderedDict):
            self.layer_names, self.layers = zip(*layers.items())
        else:
            self.layer_names = range(len(layers))
            self.layers = layers

    @symbolic
    def forward_pass_and_state(self, x):
        state = {}
        for layer in self.layers:
            x, layer_state = forward_pass_and_state(layer, x)
            state[layer]=layer_state
        return x, state

    @symbolic
    def backward_pass(self, state, grad, loss):
        assert (grad is None) != (loss is None), 'Gove me a grad xor give me a loss.'
        param_grad_pairs = []
        for layer in self.layers[::-1]:
            grad, layer_param_grad_pairs = backward_pass(layer, state[layer], grad, loss)
            loss = None
            param_grad_pairs += layer_param_grad_pairs
        return grad, param_grad_pairs

    @property
    def parameters(self):
        return [p for layer in self.layers for p in get_parameters_or_not(layer)]


class IdentityLayer(object):

    def __call__(self, x):
        return x


class SiameseNetwork(IManualBackpropLayer):
    """
    Implements:

        y = f_merge(f1(f_siamese(x1)), f2(f_siamese(x2)))

    """

    def __init__(self, f_siamese, f_merge, f1 = IdentityLayer(), f2 = IdentityLayer()):
        """
        :param f_siamese: A function or ManualBackpropLayer of the form f(
        :param f_merge:
        :return:
        """
        self.f_siamese = f_siamese
        self.f1 = f1
        self.f2 = f2
        self.f_merge = f_merge

    @symbolic
    def forward_pass_and_state(self, (x1, x2)):
        out1a, state1a = forward_pass_and_state(self.f_siamese, x1)
        out2a, state2a = forward_pass_and_state(self.f_siamese, x2)

        out1b, state1b = forward_pass_and_state(self.f1, out1a)
        out2b, state2b = forward_pass_and_state(self.f2, out2a)

        out, state_merge = forward_pass_and_state(self.f_merge, (out1b, out2b))
        return out, (state1a, state2a, state1b, state2b, state_merge)

    @symbolic
    def backward_pass(self, state, grad, loss):
        state1a, state2a, state1b, state2b, state_merge = state
        (grad_out1b, grad_out2b), merge_param_grads = backward_pass(self.f_merge, state=state_merge, grad=grad, loss=loss)
        grad_out1a, param_grads_1b = backward_pass(self.f1, state = state1b, grad=grad_out1b, loss=None)
        grad_out2a, param_grads_2b = backward_pass(self.f2, state = state2b, grad=grad_out2b, loss=None)
        grad1, param_grads_1a = backward_pass(self.f_siamese, state=state1a, grad=grad_out1a, loss=None)
        grad2, param_grads_2a = backward_pass(self.f_siamese, state=state2a, grad=grad_out2a, loss=None)

        assert all(param1 is param2 for (param1, _), (param2, _) in zip(param_grads_1a, param_grads_2a))
        param_grads_siamese = [(p1, v1+v2) for (p1, v1), (p2, v2) in zip(param_grads_1a, param_grads_2a)]
        param_grads = param_grads_siamese + param_grads_1b + param_grads_2b + merge_param_grads
        return (grad1, grad2), param_grads

    @property
    def parameters(self):
        return get_parameters_or_not(self.f_siamese) + get_parameters_or_not(self.f_merge)


class AddingLayer(IManualBackpropLayer):

    def forward_pass_and_state(self, (x1, x2)):
        return x1+x2, None

    def backward_pass(self, state, grad, loss):
        return (grad, grad), []

    @property
    def parameters(self):
        return []


@symbolic
class ConcatenationLayer(object):

    def __call__(self, (x1, x2)):
        return tt.concatenate([x1.flatten(2), x2.flatten(2)], axis=1)



class ExactBackpropLayer(IManualBackpropLayer):
    """
    Performs the function of a layer.

    (Not really useful, since you can now just feed any old function into a manual backprop net)
    """

    def __init__(self, linear_transform, nonlinearity):
        """
        linear_transform: Can be:
            A callable (e.g. FullyConnectedBridge/ConvolutionalBridge) which does a linear transform on the data.
            A numpy array - in which case it will be used to instantiate a linear transform.
        """
        if isinstance(linear_transform, np.ndarray):
            assert (linear_transform.ndim == 2 and nonlinearity!='maxout') or (linear_transform.ndim == 3 and nonlinearity=='maxout'), \
                'Your weight matrix must be 2-D (or 3-D if you have maxout units)'
            linear_transform = FullyConnectedTransform(w=linear_transform)
        if isinstance(nonlinearity, str):
            nonlinearity = get_named_activation_function(nonlinearity)
        self.linear_transform = linear_transform
        self.nonlinearity = nonlinearity

    def forward_pass_and_state(self, x):
        pre_sig = self.linear_transform(x)
        return self.nonlinearity(pre_sig), (x, pre_sig, )

    def backward_pass(self, state, grad, loss):
        x, _ = state
        if loss is None:
            y, (x, pre_sig) = self.forward_pass_and_state(x)
            dydp = tt.grad(y.sum(), wrt=pre_sig)
            # Note... we rely on the (linear-transform, pointwise-nonlinearity) design here.  We should figure out how
            # to do it more generally (maybe using tt.jacobian), or somehow making a virtual cost.
            dcdp = grad*dydp
            dcdw = x.T.dot(dcdp)  # Because I think if we did this directly for the ws we'd be in trouble
            dcdb = dcdp.sum(axis=0)
            dcdx = dcdp.dot(self.linear_transform.w.T)
            return dcdx, list(izip_equal(self.linear_transform.parameters, [dcdw, dcdb]))
        else:
            param_grads = tt.grad(loss, wrt=self.linear_transform.parameters)
            return tt.grad(loss, wrt=x), list(izip_equal(self.linear_transform.parameters, param_grads))

    @property
    def parameters(self):
        return self.linear_transform.parameters

# woooo
#fdsfdsf

# ccccc