from abc import abstractmethod
from collections import OrderedDict

import numpy as np
from artemis.general.should_be_builtins import izip_equal
from plato.core import create_constant, symbolic
from plato.interfaces.helpers import batchify_function, get_named_activation_function
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
    def __init__(self, layers, optimizer, loss, prediction_minibatch_size=None, pass_loss = True):
        """
        :param layrs:
        :param optimizer:
        :param loss:
        """
        if isinstance(layers, OrderedDict):
            self.layer_names, self.layers = zip(*layers.items())
        else:
            self.layer_names = range(len(layers))
            self.layers = layers
        self.optimizer = optimizer
        self.pass_loss = pass_loss
        self.loss = get_named_cost_function(loss) if isinstance(loss, basestring) else loss
        self.prediction_minibatch_size = prediction_minibatch_size

    @symbolic
    def predict(self, x):
        if self.prediction_minibatch_size is None:
            return self._predict_in_single_pass(x)
        else:
            return batchify_function(self._predict_in_single_pass, batch_size=self.prediction_minibatch_size)(x)

    def _predict_in_single_pass(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x

    @symbolic
    def _predict_minibatch(self, start, end, x):
        return self.predict(x[start:end], _single_pass=True)

    @symbolic
    def forward_pass_and_state(self, x):
        state = {}
        for layer in self.layers:
            if isinstance(layer, IManualBackpropLayer):
                x, layer_state = layer.forward_pass_and_state(x)
            else:
                out = layer(x)
                layer_state = (x, out)
                x = out
            state[layer]=layer_state
        return x, state

    def backward_pass(self, state, grad, loss):
        assert (grad is None) != (loss is None), 'Gove me a grad xor give me a loss.'
        layerwise_param_grad_pairs = []
        for layer in self.layers[::-1]:
            if isinstance(layer, IManualBackpropLayer):
                grad, param_grads = layer.backward_pass(state=state[layer], grad=grad, cost = loss)
            else:
                out, y = state[layer]
                grads = tt.grad(cost=loss, wrt=[out] + list(layer.parameters), known_grads={y: grad} if grad is not None else None)
                grad, param_grads = grads[0], grads[1:]
            loss = None
            layerwise_param_grad_pairs.append(list(izip_equal(layer.parameters, param_grads)))
        if isinstance(self.optimizer, IGradientOptimizer):
            all_params, all_param_grads = zip(*[(p, g) for layer_pairs in layerwise_param_grad_pairs for p, g in layer_pairs])
            self.optimizer.update_from_gradients(parameters=all_params, gradients=all_param_grads)
        elif isinstance(self.optimizer, (list, tuple)):
            for optimizer, layer_pairs in izip_equal(self.optimizer, layerwise_param_grad_pairs):
                params, grads = zip(*layer_pairs)
                optimizer.update_from_gradients(parameters=params, gradients=grads)

    @symbolic
    def train(self, x, y):
        out, state = self.forward_pass_and_state(x)
        loss = self.loss(out, y)
        if self.pass_loss:
            grad = None
        else:
            grad = tt.grad(loss, wrt=out)
            loss = None
        self.backward_pass(state, grad, loss)

    @property
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters]


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
    def backward_pass(self, state, grad, cost):
        """
        :param state: The list of state variables you returned in forward_pass_and_state
        :param grad: The incoming gradient
        :return: The outgoing gradient
        """
        raise NotImplementedError()


class ManualBackPropChain(IManualBackpropLayer):

    def __init__(self, layers):
        if isinstance(layers, OrderedDict):
            self.layer_names, self.layers = zip(*layers.items())
        else:
            self.layer_names = range(len(layers))
            self.layers = layers





class ExactBackpropLayer(IManualBackpropLayer):
    """
    Performs the function of a layer.
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

    def backward_pass(self, state, grad, cost):
        x, _ = state
        if cost is None:
            y, (x, pre_sig) = self.forward_pass_and_state(x)
            dydp = tt.grad(y.sum(), wrt=pre_sig)
            # Note... we rely on the (linear-transform, pointwise-nonlinearity) design here.  We should figure out how
            # to do it more generally (maybe using tt.jacobian), or somehow making a virtual cost.
            dcdp = grad*dydp
            dcdw = x.T.dot(dcdp)  # Because I think if we did this directly for the ws we'd be in trouble
            dcdb = dcdp.sum(axis=0)
            dcdx = dcdp.dot(self.linear_transform.w.T)
            return dcdx, [dcdw, dcdb]
        else:
            return tt.grad(cost, wrt=x), tt.grad(cost, wrt=self.linear_transform.parameters)

    @property
    def parameters(self):
        return self.linear_transform.parameters
