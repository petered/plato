from collections import namedtuple
from plato.interfaces.decorators import symbolic_standard, symbolic_stateless, find_shared_ancestors
from plato.interfaces.interfaces import IParameterized, IFreeEnergy
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
        :param input_size: An integer indicating the size of the input layer
        :param hidden_activation: A string or list of strings indicating the type of each hidden layer.
            {'sig', 'tanh', 'rect-lin', 'lin', 'softmax'}
        :param output_activation: A string (see above) identifying the activation function for the output layer
        :param w_init_mag: Standard-Deviation of the gaussian-distributed initial weights
        :param rng: The Random number generator to use for initial weight values.
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
    """
    An element that applies a nonlinearity to its inputs.  If multiple vectors
    of input feed into a layer, they are summed before the nonlinearity.
    """

    def __init__(self, activation_fcn):
        """
        :param activation_fcn: A string identifying the type of activation function.
            {'sig', 'lin', 'tanh', 'rect-lin', 'softmax'}
        """
        if isinstance(activation_fcn, str):
            activation_fcn = {
                'sig': tt.nnet.sigmoid,
                'lin': lambda x: x,
                'tanh': tt.tanh,
                'rect-lin': lambda x: tt.maximum(0, x),
                'softmax': lambda x: tt.nnet.softmax(x)
            }[activation_fcn]
        self._activation_fcn = activation_fcn

    def __call__(self, *input_currents):
        summed_current = sum(input_currents)
        out = self._activation_fcn(summed_current)
        return out


@symbolic_stateless
class StochasticLayer(IParameterized, IFreeEnergy):
    """
    A stochastic layer, which can also be called without the stochastic component
    (see smooth method).  These are building blocks in RBMs.
    """

    def __init__(self, activation_fcn, rng = None, shape = None):
        """
        :param activation_fcn: A string identifying the type of activation function.
            {'bernoulli', 'gaussian', 'adaptive_gaussian', 'rect-lin'}
        :param rng: Numpy random number generator for the stochastic component
        :param shape: Optionally, reshape the output to this shape.
        """
        rng = RandomStreams(rng.randint(1e9) if rng is not None else None)
        self._smooth_activation_fcn, self._stochastic_activation_fcn, self._free_energy_fcn, self._params = \
            self._stochastic_layer_name_to_functions(activation_fcn, rng)
        self._shape = shape

    def __call__(self, *input_currents):
        summed_current = sum(input_currents)
        out = self._stochastic_activation_fcn(summed_current)
        if self._shape is not None:
            out = out.reshape(self._shape)
        return out

    def smooth(self, *input_currents):
        summed_current = sum(input_currents)
        out = self._smooth_activation_fcn(summed_current)
        if self._shape is not None:
            out = out.reshape(self._shape)
        return out

    @property
    def parameters(self):
        return self._params

    def free_energy(self, x):
        if self._free_energy_fcn is None:
            raise NotImplementedError('No Free energy function implemented for this layer type!')
        else:
            return self._free_energy_fcn(x)

    @staticmethod
    def _stochastic_layer_name_to_functions(activation_type, rng):
        params = []
        if activation_type == 'bernoulli':
            smooth_activation_fcn = lambda x: tt.nnet.sigmoid(x)
            stochastic_activation_fcn = lambda x: rng.binomial(p=tt.nnet.sigmoid(x), dtype = theano.config.floatX)
            free_energy_fcn = lambda x: -tt.nnet.softplus(x).sum(axis = 1)
        elif activation_type == 'gaussian':
            smooth_activation_fcn = lambda x: x
            stochastic_activation_fcn = lambda x: rng.normal(avg = x, std = 1)
            free_energy_fcn = None
        elif activation_type == 'adaptive_gaussian':
            smooth_activation_fcn = lambda x: x
            sigma = theano.shared(1, name = 'sigma', dtype = theano.config.floatX)
            stochastic_activation_fcn = lambda x: rng.normal(avg = x, std = sigma)
            free_energy_fcn = None
            params.append(sigma)
        elif activation_type == 'rect-lin':
            smooth_activation_fcn = lambda x: np.maximum(0, x)
            stochastic_activation_fcn = lambda x: np.maximum(0, )
            free_energy_fcn = None
        else:
            raise Exception('Unknown activation type: "%s"' (activation_type, ))

        return smooth_activation_fcn, stochastic_activation_fcn, free_energy_fcn, params


@symbolic_stateless
class FullyConnectedBridge(IParameterized, IFreeEnergy):
    """
    An element which multiplies the input by some weight matrix w and adds a bias.
    """

    def __init__(self, w, b = None, b_rev = None):
        if isinstance(w, np.ndarray):  # Initialize from array
            assert w.ndim == 2, 'w must be a 2-d matrix of initial weights'
            self._w = self._w_param = theano.shared(w, name = 'w', borrow = True, allow_downcast=True)
            n_in, n_out = w.shape
        else:  # Initialize from a variable
            self._w = w
            self._w_param, = find_shared_ancestors(w)
            assert self._w_param.get_value().ndim == 2
            n_in, n_out = w.shape if w is self._w_param else self._w_param.shape[::-1]

        if b is None:
            b = np.zeros(n_out, dtype = theano.config.floatX)
        if b_rev is None:
            b_rev = np.zeros(n_in, dtype = theano.config.floatX)
        assert b.ndim == b_rev.ndim == 1, 'b must be a vector representing the inital biases'
        self._b = theano.shared(b, 'b', borrow = True, allow_downcast=True)
        self._b_rev = theano.shared(b_rev, 'b_rev', borrow = True, allow_downcast=True)

    def __call__(self, x):
        y = x.flatten(2).dot(self._w) + self._b
        return y

    @property
    def parameters(self):
        return [self._w_param, self._b, self._b_rev]

    def reverse(self, y):
        return y.flatten(2).dot(self._w.T)+self._b_rev

    def free_energy(self, visible):
        return -visible.flatten(2).dot(self._b_rev)
