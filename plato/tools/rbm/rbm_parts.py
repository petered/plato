from plato.core import symbolic_simple, initialize_param, create_shared_variable
from plato.interfaces.interfaces import IParameterized, IFreeEnergy
import theano
import theano.tensor as tt
from theano.tensor.shared_randomstreams import RandomStreams

__author__ = 'peter'


@symbolic_simple
class FullyConnectedBridge(IParameterized, IFreeEnergy):
    """
    An element which multiplies the input by some weight matrix w and adds a bias.
    """

    def __init__(self, w, b = 0, b_rev = None, use_bias = True):
        """
        :param w: Initial weight value.  Can be:
            - A numpy array, in which case a shared variable is instantiated from this data.
            - A symbolic variable that is either a shared variabe or descended from a shared variable.
              This is used when there are shared parameters.
        :param b: Can be:
            - A numpy vector representing the initial bias on the hidden layer, where len(b) = w.shape[1]
            - A scaler, which just initializes the full vector to this value
        :param b_rev: Can be:
            - A numpy vector representing the initial bias on the visible layer, where len(b) = w.shape[0]
            - A scaler, which just initializes the full vector to this value
            - None, in which case b_rev is not created (for instance in an MLP).
        """
        self.w = create_shared_variable(w, name = 'w')
        self.b = create_shared_variable(b, shape = w.shape[1], name = 'b') if use_bias else None
        self.b_rev = create_shared_variable(b_rev, shape = w.shape[0], name = 'b_rev') if use_bias else None
        self._use_bias = use_bias

    def __call__(self, x):
        current = x.flatten(2).dot(self.w)
        y = current + self.b if self._use_bias else current
        return y

    @property
    def parameters(self):
        return [self.w, self.b, self.b_rev] if self._use_bias else [self.w]

    def reverse(self, y):
        current = y.flatten(2).dot(self.w.T)
        x = current + self.b_rev if self._use_bias else current
        return x

    def free_energy(self, visible):
        return -visible.flatten(2).dot(self.b_rev) if self._use_bias else 0


@symbolic_simple
class ConvolutionalBridge(IParameterized, IFreeEnergy):

    def __init__(self, w, b=0, b_rev=None, stride = (1, 1)):
        self._w, w_params, w_shape = initialize_param(w, shape = (None, None, None, None), name = 'w')
        self._b, b_params, b_shape = initialize_param(b, shape = w_shape[0], name = 'b')
        self._b_rev, b_rev_params, b_rev_shape = initialize_param(b_rev, shape = w_shape[1], name = 'b_rev')
        self._params = w_params+b_params+b_rev_params
        self._stride = stride

    def __call__(self, x):
        y = tt.nnet.conv2d(x, self._w, border_mode='valid', subsample = self._stride) + self._b.dimshuffle('x', 0, 'x', 'x')
        return y

    @property
    def parameters(self):
        return self._params

    def reverse(self, y):

        assert self._stride == (1, 1), 'Only support single-step strides for now...'
        # But there's this approach... https://groups.google.com/forum/#!topic/theano-users/Xw4d00iV4yk
        return tt.nnet.conv2d(y, self._w.dimshuffle(1, 0, 2, 3)[:, :, ::-1, ::-1], border_mode='full') + self._b_rev.dimshuffle('x', 0, 'x', 'x')

    def free_energy(self, visible):
        return -tt.sum(visible*self._b_rev.dimshuffle('x', 0, 'x', 'x'), axis = (2, 3))


@symbolic_simple
class StochasticNonlinearity(IParameterized, IFreeEnergy):
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
        self.activation_fcn = activation_fcn
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
        elif activation_type in ('rect-lin', 'relu'):
            smooth_activation_fcn = lambda x: tt.maximum(0, x)
            stochastic_activation_fcn = lambda x: tt.maximum(0, x+rng.normal(avg=0, std=tt.sqrt(tt.nnet.sigmoid(x)), size = x.tag.test_value.shape))
            free_energy_fcn = lambda x: -tt.nnet.softplus(x).sum(axis = 1)
        else:
            raise Exception('Unknown activation type: "%s"' (activation_type, ))

        return smooth_activation_fcn, stochastic_activation_fcn, free_energy_fcn, params