from plato.interfaces.decorators import symbolic_standard, symbolic_stateless, find_shared_ancestors
from plato.interfaces.interfaces import IParameterized
from theano.gof.graph import Variable
import theano.tensor as tt
import theano
import numpy as np
from plato.tools.linking import Chain
from theano.tensor.shared_randomstreams import RandomStreams


@symbolic_stateless
class MultiLayerPerceptron(IParameterized):
    """
    A Multi-Layer Perceptron
    """

    def __init__(self, layer_sizes, input_size, hidden_activation = 'sig', output_activation = 'sig', w_init_mag = 0.1, rng = None):
        """
        :param layer_sizes: A list indicating the sizes of each layer.
        :param hidden_types: A string or list of strings indicating the type of each hidden layer.
        :param output_type: A string indicating the type of the output layer
        :return:
        """
        if rng is None:
            rng = np.random.RandomState()

        all_layer_sizes = [input_size]+layer_sizes
        all_layer_activations = [hidden_activation] * (len(layer_sizes)-1) + [output_activation]
        processors = sum([[
             FullyConnectedBridge(w = w_init_mag*rng.randn(pre_size, post_size)),
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
                'sig': tt.nnet.sigmoid,
                'lin': lambda x: x,
                'tanh': tt.tanh,
                'rect-lin': lambda x: tt.maximum(0, x),
            }[activation_fcn]
        self._activation_fcn = activation_fcn

    def __call__(self, *input_currents):
        summed_current = sum(input_currents)
        out = self._activation_fcn(summed_current)
        return out


@symbolic_stateless
class StochasticLayer(IParameterized):

    def __init__(self, activation_type, rng):

        self._rng = RandomStreams(rng.randint(1e9))
        if isinstance(activation_type, str):
            params = []
            if activation_type == 'bernoulli':
                smooth_activation_fcn = lambda x: tt.nnet.sigmoid(x)
                stochastic_activation_fcn = lambda x: self._rng.binomial(tt.nnet.sigmoid(x))
            elif activation_type == 'gaussian':
                smooth_activation_fcn = lambda x: x
                stochastic_activation_fcn = lambda x: self._rng.normal(avg = x, std = 1)
            elif self._stochastic_activation_fcn == 'adaptive_gaussian':
                smooth_activation_fcn = lambda x: x
                sigma = theano.shared(1, name = 'sigma', dtype = theano.config.floatX)
                stochastic_activation_fcn = lambda x: self._rng.normal(avg = x, std = sigma)
                self._params.append(sigma)
            elif self._activation_fcn == 'rect-lin':
                smooth_activation_fcn = lambda x: np.maximum(0, x)
                stochastic_activation_fcn = lambda x: np.maximum(0, )
            else:
                raise Exception('Unknown activation type: "%s"' (activation_type, ))
        else:
            smooth_activation_fcn, stochastic_activation_fcn, params = activation_type
        self._smooth_activation_fcn = smooth_activation_fcn
        self._stochastic_activation_fcn = stochastic_activation_fcn
        self._params = params

    def __call__(self, *input_currents):
        summed_current = sum(input_currents)
        out = self._stochastic_activation_fcn(summed_current)
        return out

    def smooth(self, *input_currents):
        summed_current = sum(input_currents)
        out = self._smooth_activation_fcn(summed_current)
        return out

    @property
    def parameters(self):
        return self._params


@symbolic_stateless
class FullyConnectedBridge(IParameterized):

    def __init__(self, w, b = None):
        if isinstance(w, np.ndarray):  # Initialize from array
            assert w.ndim == 2, 'w must be a 2-d matrix of initial weights'
            self._w = self._w_param = theano.shared(w, 'w', borrow = True, allow_downcast=True, dtype = theano.config.floatX)
        else:  # Initialize from a variable
            self._w = w
            self._w_param, = find_shared_ancestors(w)
            assert self._w_param.get_value().ndim == 2
        if b is None:
            b = np.zeros(self._w_param.shape[1 if self._w is self._w_param else ])
        assert b.ndim == 1, 'b must be a vector representing the inital biases'
        self._b = theano.shared(b, 'b', borrow = True, allow_downcast=True)

    def __call__(self, x):
        y = tt.dot(x.flatten(2), self._w) + self._b
        return y

    @property
    def parameters(self):
        return [self._w_param, self._b]

    def get_back_bridge(self):
        return FullyConnectedBridge(w = self._w.T)
