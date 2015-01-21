from plato.interfaces.decorators import symbolic_standard, symbolic_stateless
from plato.interfaces.interfaces import IParameterized
import theano.tensor as ts
import theano
import numpy as np
from plato.tools.linking import Chain


@symbolic_stateless
class MultiLayerPerceptron(IParameterized):
    """
    A Multi-Layer Perceptron
    """

    def __init__(self, layer_sizes, input_size, hidden_activation = 'sig', output_activation = 'sig', w_init_mag = 0.1):
        """
        :param layer_sizes: A list indicating the sizes of each layer.
        :param hidden_types: A string or list of strings indicating the type of each hidden layer.
        :param output_type: A string indicating the type of the output layer
        :return:
        """

        all_layer_sizes = [input_size]+layer_sizes
        all_layer_activations = [hidden_activation] * (len(layer_sizes)-1) + [output_activation]
        processors = sum([[
             FullyConnectedBridge(w = w_init_mag*np.random.randn(pre_size, post_size)),
             Layer(activation_fcn)
             ] for (pre_size, post_size), activation_fcn in zip(zip(all_layer_sizes[:-1], all_layer_sizes[1:]), all_layer_activations)
             ], [])
        self._chain = Chain(*processors)

    def __call__(self, x):
        return self._chain.symbolic_stateless(x)

    @property
    def parameters(self):
        return self._chain.parameters


@symbolic_stateless
class Layer(object):

    def __init__(self, activation_fcn):
        if isinstance(activation_fcn, str):
            activation_fcn = {
                'sig': ts.nnet.sigmoid,
                'lin': lambda x: x,
                'tanh': ts.tanh,
                'rect-lin': lambda x: ts.maximum(0, x),
            }[activation_fcn]
        self._activation_fcn = activation_fcn

    def __call__(self, *input_currents):
        summed_current = sum(input_currents)
        out = self._activation_fcn(summed_current)
        return out


@symbolic_stateless
class FullyConnectedBridge(IParameterized):

    def __init__(self, w, b = None):
        assert w.ndim == 2, 'w must be a 2-d matrix of initial weights'
        if b is None:
            b = np.zeros(w.shape[1])
        assert b.ndim == 1, 'b must be a vector representing the inital biases'
        self._w = theano.shared(w, 'w')
        self._b = theano.shared(b, 'b')

    def __call__(self, x):
        y = ts.dot(x.flatten(2), self._w) + self._b
        return y

    @property
    def parameters(self):
        return [self._w, self._b]
