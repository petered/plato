from plato.interfaces.decorators import symbolic_stateless, find_shared_ancestors
from plato.interfaces.interfaces import IParameterized, IFreeEnergy
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

    def __init__(self, layer_sizes, input_size, hidden_activation = 'sig', output_activation = 'sig',
            normalize_minibatch = False, scale_param = False, w_init = lambda n_in, n_out: 0.1*np.random.randn(n_in, n_out)):
        """
        :param layer_sizes: A list indicating the sizes of each layer.
        :param input_size: An integer indicating the size of the input layer
        :param hidden_activation: A string or list of strings indicating the type of each hidden layer.
            {'sig', 'tanh', 'rect-lin', 'lin', 'softmax'}
        :param output_activation: A string (see above) identifying the activation function for the output layer
        :param w_init: A function which, given input dims, output dims, return
        """

        all_layer_sizes = [input_size]+layer_sizes
        all_layer_activations = [hidden_activation] * (len(layer_sizes)-1) + [output_activation]
        processors = sum([[
             FullyConnectedBridge(w = w_init(pre_size, post_size), normalize_minibatch=normalize_minibatch, scale = scale_param),
             Layer(activation_fcn)
             ] for (pre_size, post_size), activation_fcn in zip(zip(all_layer_sizes[:-1], all_layer_sizes[1:]), all_layer_activations)
             ], [])

        self._chain = Chain(*processors)

    def __call__(self, x):
        return self._chain.symbolic_stateless(x)

    @property
    def parameters(self):
        return self._chain.parameters


def normal_w_init(mag, seed = None):
    rng = np.random.RandomState(seed)
    return lambda n_in, n_out: mag * rng.randn(n_in, n_out)


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
                'softmax': lambda x: tt.nnet.softmax(x),
                'exp': lambda x: tt.exp(x)
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
            smooth_activation_fcn = lambda x: tt.maximum(0, x)
            stochastic_activation_fcn = lambda x: tt.maximum(0, x+rng.normal(avg=0, std=.1))
            free_energy_fcn = lambda x: -tt.nnet.softplus(x).sum(axis = 1)
        else:
            raise Exception('Unknown activation type: "%s"' (activation_type, ))

        return smooth_activation_fcn, stochastic_activation_fcn, free_energy_fcn, params



@symbolic_stateless
class FullyConnectedBridge(IParameterized, IFreeEnergy):
    """
    An element which multiplies the input by some weight matrix w and adds a bias.
    """

    def __init__(self, w, b = 0, b_rev = None, scale = False, normalize_minibatch = False):
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
        self._w, w_params, w_shape = _initialize_param(w, shape = (None, None), name = 'w')
        self._b, b_params, b_shape = _initialize_param(b, shape = w_shape[1], name = 'b')
        self._b_rev, b_rev_params, b_rev_shape = _initialize_param(b_rev, shape = w_shape[0], name = 'b_rev')
        self._log_scale, log_scale_params, log_scale_shape = _initialize_param(0 if scale else None, shape = w.shape[1], name = 'log_scale')
        self._params = w_params+b_params+b_rev_params+log_scale_params
        self._normalize_minibatch = normalize_minibatch

    def __call__(self, x):
        current = x.flatten(2).dot(self._w)

        if self._normalize_minibatch:
            current = (current - current.mean(axis = 0, keepdims = True)) / (current.std(axis = 0, keepdims = True) + 1e-9)

        if self._log_scale is not None:
            current = current * tt.exp(self._log_scale)

        y = current + self._b
        return y

    @property
    def parameters(self):
        return self._params

    def reverse(self, y):
        assert self._b_rev is not None, 'You are calling reverse on this bridge, but you failed to specify b_rev.'
        assert not self._normalize_minibatch, "Don't really know about this case..."
        return y.flatten(2).dot(self._w.T)+self._b_rev

    def free_energy(self, visible):
        return -visible.flatten(2).dot(self._b_rev)


@symbolic_stateless
class ConvolutionalBridge(IParameterized, IFreeEnergy):

    def __init__(self, w, b=0, b_rev=None, stride = (1, 1)):
        self._w, w_params, w_shape = _initialize_param(w, shape = (None, None, None, None), name = 'w')
        self._b, b_params, b_shape = _initialize_param(b, shape = w_shape[0], name = 'b')
        self._b_rev, b_rev_params, b_rev_shape = _initialize_param(b_rev, shape = w_shape[1], name = 'b_rev')
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


def _initialize_param(initial_value, shape = None, name = None, cast_floats_to_floatX = True):
    """
    Takes care of the common stuff associated with initializing a parameter.  There are a few ways you may want to
    instantiate a parameter:
    - With a numpy array, in which case you'll want to make sure it's the appropriate shape.
    - With a scalar, in which case you just want a scalar shared variable.
    - With a scalar and a shape, in which case you want an array of that shape filled with the value of the scalar.
    - With a symbolic variable descenting from some other shared variable - this is the case when you want to tie
      parameters together, or make the bias be the result of a previous computation, etc.
    - With None, which we take to mean that this was an optional variable that should not be included, so return None
      for variable, param, and shape.

    :param initial_value: An array, scalar, or symbolic variable.:
    :param shape: The shape that the variable should have.  None if it is already fully specified by initial_value.
        If shape is a tuple, elements of shape s can be:
        - integers: In which case, they mean (the dimension of in this direction shall be <s>
        - None: In which case, the initial_value must be defined as an array, and the array may have any shape along this axis.
    :param name: Optionally, the name for the shared variable.
    :return: (variable, param): Variable is the shared variable, and param is the associated parameter.  If you
        instantiated with scalar or ndarray, variable and param will be the same object.
    """

    if isinstance(shape, int):
        shape = (shape, )

    typecast = lambda x: x.astype(theano.config.floatX) if cast_floats_to_floatX and x.dtype=='float' else x

    if np.isscalar(initial_value):
        if shape is None:
            initial_value = np.array(initial_value)
        else:
            initial_value = np.zeros(shape)+initial_value
        initial_value = typecast(initial_value)
    if isinstance(initial_value, np.ndarray):
        assert_compatible_shape(initial_value.shape, shape, name = name)
        variable = theano.shared(typecast(initial_value), name = name, borrow = True, allow_downcast=True)
        params = [variable]
        variable_shape = initial_value.shape
    elif initial_value is Variable:
        assert name is None, "Can't give name '%s' to an already-existing symbolic variable" % (name, )
        params = find_shared_ancestors(initial_value)
        # Note to self: possibly remove this constraint for things like factored weight matrices?
        if len(params)==1 and initial_value is params[0]:
            variable_shape = initial_value.get_value().shape
            assert_compatible_shape(variable_shape, shape, name = name)
        else:
            raise NotImplementedError("Can't yet get variable shape from base-params, though this can be done cheaply in "
                'Theano by compiling a function wholse input is the params and whose output is the shape.')
    elif initial_value is None:
        variable = None
        params = []
        variable_shape = None
    else:
        raise Exception("Don't know how to instantiate variable from %s" % initial_value)
    return variable, params, variable_shape


def assert_compatible_shape(actual_shape, desired_shape, name = None):
    """
    Return a boolean indicating whether the actual shape is compatible with the desired shape.  "None" serves as a wildcard.
    :param actual_shape: A tuple<int>
    :param desired_shape: A tuple<int> or None
    :return: A boolean

    examples (actual_desired, desired_shape : result)
    (1, 2), None: True        # Because None matches everything
    (1, 2), (1, None): True   # Because they have the same length, 1 matches 1 and 2 matches None
    (1, 2), (1, 3): False     # Because 2 != 3
    (1, 2, 1), (1, 2): False  # Because they have different lengths.
    """
    return desired_shape is None or len(actual_shape) == len(desired_shape) and all(ds is None or s==ds for s, ds in zip(actual_shape, desired_shape)), \
        "Actual shape %s%s did not correspond to specified shape, %s" % (actual_shape, '' if name is None else ' of %s' %(name, ), desired_shape)
