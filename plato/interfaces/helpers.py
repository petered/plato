import numpy as np
from plato.interfaces.decorators import find_shared_ancestors
from plato.tools.common.basic import softmax
import theano
from theano import Variable
from theano.sandbox.cuda.rng_curand import CURAND_RandomStreams
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano.tensor.shared_randomstreams import RandomStreams
import theano.tensor as tt

__author__ = 'peter'


class MRG_RandomStreams_ext(MRG_RandomStreams):
    """
    Add some basic methods to MRG_randomstreams
    """

    def choice(self, a=2, p=None, size = None, replace = True):

        if isinstance(a, int):
            options = tt.arange(a)
        elif isinstance(a, (np.ndarray, tuple, list)):
            options = tt.constant(a)
        else:
            options=a

        if replace is False:
            assert p is None, 'Only supports uniform distributions right now'
            assert isinstance(size, int), 'Only supports vectors right now'
            assert options.ndim == 1, 'Only supports vector values'
            ixs = tt.argsort(self.uniform(size = options.shape))
            return options[ixs[:size]]
        else:
            assert len(a) == 2, 'Only support Bernoulli choices for now'
            assert size is not None, 'Please specify size!'
            ix = (self.uniform(size = size) < p).astype('int32')
            return options[ix]


def get_theano_rng(seed, rngtype = 'mrg'):
    """
    Helper for getting a theano random number generator.  How this is started depends on the form
    of the seed.

    :param seed: Can be:
        - An integer, in which case the random number generator is seeded with this..
        - None, in which case a random seed is chosen
        - A numpy random number generator, in which case we randomly select a seed from this.
        - A theano random number generator, in which case we just pass it through.
    :param rngtype: The type of random number generator to use.  Options are:
        - 'default': The default theano type (which seems to be slow)
        - 'mrg': The
    :return:
    """

    stream_types = {
        'mrg': MRG_RandomStreams_ext,
        'mrg-old': MRG_RandomStreams,
        'default': RandomStreams,
        'cuda': CURAND_RandomStreams
    }
    rng_con = stream_types[rngtype]

    if isinstance(seed, np.random.RandomState):
        return rng_con(seed.randint(1e9))
    elif isinstance(seed, int):
        return rng_con(seed)
    elif seed is None:
        return rng_con(np.random.randint(1e9))
    elif isinstance(seed, tuple(stream_types.values())):
        return seed
    else:
        raise Exception("Can't initialize a random number generator with %s" % (seed, ))


def initialize_param(initial_value, shape = None, name = None, cast_floats_to_floatX = True):
    """
    Takes care of the common stuff associated with initializing a parameter.  There are a few ways you may want to
    instantiate a parameter:
    - With a numpy array, in which case you'll want to make sure it's the appropriate shape.
    - With a scalar, in which case you just want a scalar shared variable.
    - With a scalar and a shape, in which case you want an array of that shape filled with the value of the scalar.
    - With a symbolic variable descenting from some other shared variable - this is the case when you want to tie
      parameters together, or make the bias be the result of a previous computation, etc.
    - With a function and shape, in which case the function should return an initial numpy array given the shape.
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

    typecast = lambda x: x.astype(theano.config.floatX) if cast_floats_to_floatX and x.dtype in ('float', 'float64', 'float32') else x

    if np.isscalar(initial_value):
        if shape is None:
            initial_value = np.array(initial_value)
        else:
            initial_value = np.zeros(shape)+initial_value
        initial_value = typecast(initial_value)
    elif hasattr(initial_value, '__call__'):
        assert shape is not None, "If you initialize with a function, you must provide a shape."
        initial_value = initial_value(shape)

    if isinstance(initial_value, np.ndarray):
        assert_compatible_shape(initial_value.shape, shape, name = name)
        variable = theano.shared(typecast(initial_value), name = name, borrow = True, allow_downcast=True)
        params = [variable]
        variable_shape = initial_value.shape
    elif isinstance(initial_value, Variable):
        assert name is None or initial_value.name == name, "Can't give name '%s' to an already-existing symbolic variable" % (name, )
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


def create_shared_variable(initializer_fcn, shape = None, name = None, cast_floats_to_floatX = True):
    """
    :param initializer_fcn: Can be:
        - An array.  It may be cast to floatX.  It's verified with shape if shape is provided
        - A function which takes the shape and turns it into the array.
        - A scalar, in which case it's broadcase over shape.
    :param shape: Either a tuple or an integer
    :return: A shared variable, containing the numpy array returned by the initializer.
    """
    shared_var, _, _ = initialize_param(initializer_fcn, shape = shape, name = name, cast_floats_to_floatX=cast_floats_to_floatX)
    return shared_var


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


normalize= lambda x, axis = None: x/(x.sum(axis=axis, keepdims = True) + 1e-9)

normalize_safely= lambda x, axis = None, degree = 1: x/((x**degree).sum(axis=axis, keepdims = True) + 1)**(1./degree)


def get_named_activation_function(activation_name):
    return {
            'softmax': lambda x: softmax(x, axis = -1),
            'sigm': tt.nnet.sigmoid,
            'sig': tt.nnet.sigmoid,
            'tanh': tt.tanh,
            'lin': lambda x: x,
            'exp': lambda x: tt.exp(x),
            'relu': lambda x: tt.maximum(x, 0),
            'rect-lin': lambda x: tt.maximum(0, x),
            'linear': lambda x: x,
            'softplus': lambda x: tt.nnet.softplus(x),
            'norm-relu': lambda x: normalize(tt.maximum(x, 0), axis = -1),
            'safenorm-relu': lambda x: normalize_safely(tt.maximum(x, 0), axis = -1),
            'balanced-relu': lambda x: tt.maximum(x, 0)*(2*(tt.arange(x.shape[-1]) % 2)-1),  # Glorot et al.  Deep Sparse Rectifier Networks
            'prenorm-relu': lambda x: tt.maximum(normalize_safely(x, axis = -1, degree = 2), 0),
            'linear': lambda x: x
            }[activation_name]
