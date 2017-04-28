import numpy as np
from plato.core import symbolic_simple, add_update, create_shared_variable, symbolic
from plato.interfaces.interfaces import IParameterized
import theano
from theano.compile.sharedvalue import SharedVariable
from theano.ifelse import ifelse
from theano.sandbox.cuda.rng_curand import CURAND_RandomStreams
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano.tensor.shared_randomstreams import RandomStreams
import theano.tensor as tt
from theano.tensor.sharedvar import TensorSharedVariable
from theano.tensor.var import TensorVariable

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


normalize= lambda x, axis = None: x/(x.sum(axis=axis, keepdims = True) + 1e-9)

normalize_safely= lambda x, axis = None, degree = 1: x/((x**degree).sum(axis=axis, keepdims = True) + 1)**(1./degree)


def softmax(x, axis=1):
    # Slightly more general than theano's softmax, in that it works on arbitrarily shaped arrays
    e_x = tt.exp(x - x.max(axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)


def relu(x):
    return tt.maximum(x, 0)

def identity(x):
    return x


_act_funcs = {
    'softmax': softmax,
    'sigm': tt.nnet.sigmoid,
    'sig': tt.nnet.sigmoid,
    'd_sigm': lambda x: tt.nnet.sigmoid(x)-tt.nnet.sigmoid(-x),
    'tanh': tt.tanh,
    'sech2': lambda x: (4*tt.cosh(x)**2)/(tt.cosh(2*x)+1)**2,
    'lin': identity,
    'const': tt.ones_like,
    'step': lambda x: tt.switch(x<0, 0., 1.),
    'exp': tt.exp,
    'relu': relu,
    'rect-lin': relu,
    'softmax-last': tt.nnet.softmax,
    'softplus': tt.nnet.softplus,
    'norm-relu': lambda x: normalize(tt.maximum(x, 0), axis = -1),
    'safenorm-relu': lambda x: normalize_safely(tt.maximum(x, 0), axis = -1),
    'balanced-relu': lambda x: tt.maximum(x, 0)*(2*(tt.arange(x.shape[-1]) % 2)-1),  # Glorot et al.  Deep Sparse Rectifier Networks
    'prenorm-relu': lambda x: tt.maximum(normalize_safely(x, axis = -1, degree = 2), 0),
    'linear': identity,
    'leaky-relu-0.01': lambda x: tt.maximum(0.01*x, x),
    'maxout': lambda x: tt.max(x, axis=1),  # We expect (n_samples, n_maps, n_dims) data and flatten to (n_samples, n_dims)
     }


def get_named_activation_function(activation_name):
    fcn = _act_funcs[activation_name]
    return symbolic_simple(fcn)


def get_named_activation_function_derivative(activation_name):
    fcn = _act_funcs[{
        'sigm': 'd_sigm',
        'relu': 'step',
        'linear': 'const',
        'lin': 'const',
        'softplus': 'sigm',
        'tanh': 'sech2',
        }[activation_name]]
    return symbolic_simple(fcn)



#
# def get_named_activation_derivative(activation_name):
#     fcn = {}
#

def compute_activation(x, activation_name):
    return get_named_activation_function(activation_name)(x)


def get_parameters_or_not(module):
    """ Return parameters if the given module is an IParameterized object, else return an empty list."""
    return module.parameters if isinstance(module, IParameterized) else []


@symbolic_simple
def batch_normalize(x):
    return (x - x.mean(axis = 0, keepdims = True)) / (x.std(axis = 0, keepdims = True) + 1e-9)


@symbolic_simple
class SlowBatchNormalize(object):
    """
    Keeps a running mean and standard deviation, and normalizes the incoming data according to these.
    This can be useful if you want to do something similar to minibatch-normalization, but without having
    the batch-size tied to the normalization range.
    """

    def __init__(self, half_life):
        self.decay_constant = np.exp(-np.log(2)/half_life).astype(theano.config.floatX)

    def __call__(self, x):
        # x should have
        assert x.ishape[0]==1, "This method only works for minibatches of size 1, but you used a minibatch of size: %s" % (x.tag.test_value.shape[0])
        running_mean = create_shared_variable(np.zeros(x.tag.test_value.shape[1:]))
        running_mean_sq = create_shared_variable(np.zeros(x.tag.test_value.shape[1:]))
        new_running_mean = running_mean * self.decay_constant + x[0] * (1-self.decay_constant).astype(theano.config.floatX)
        new_running_mean_sq = running_mean_sq * self.decay_constant + (x[0]**2) * (1-self.decay_constant).astype(theano.config.floatX)
        add_update(running_mean, new_running_mean)
        add_update(running_mean_sq, new_running_mean_sq)
        running_std = tt.sqrt((new_running_mean_sq - new_running_mean**2))
        return (x - running_mean)/(running_std+1e-7)


@symbolic_simple
class SlowBatchCenter(object):
    """
    Keeps a running mean and standard deviation, and normalizes the incoming data according to these.
    This can be useful if you want to do something similar to minibatch-normalization, but without having
    the batch-size tied to the normalization range.
    """

    def __init__(self, half_life):
        self.decay_constant = np.exp(-np.log(2)/half_life).astype(theano.config.floatX)

    def __call__(self, x):
        # x should have
        assert x.ishape[0]==1, "This method only works for minibatches of size 1, but you used a minibatch of size: %s" % (x.tag.test_value.shape[0])
        running_mean = create_shared_variable(np.zeros(x.tag.test_value.shape[1:]))
        new_running_mean = running_mean * self.decay_constant + x[0] * (1-self.decay_constant).astype(theano.config.floatX)
        add_update(running_mean, new_running_mean)
        return x - running_mean


def batchify_function(fcn, batch_size):
    """
    Given a symbolic function, transform it so that computes its input in a sequence of minibatches, instead of in
    one go.  This can be useful when:
        - You want to perform an operation on a large array but don't have enough memory to do it all in one go
        - Your function has state, and you want to update it with each step.

    *NOTE: Currently this only works when the length of your arguments evently divides into the batch size

    :param fcn: A symbolic function of the form out = f(*args)
    :param args: A list of arguments.  All arguments must have the same arg.shape[0]
    :param batch_size: An integer indicating the size of the batches in which you'd like to process your data.
    :return:
    """

    @symbolic
    def batch_function(*args):
        start_ixs = tt.arange(0, args[0].shape[0], batch_size)
        @symbolic
        def process_batch(start_ix, end_ix):
            return fcn(*[arg[start_ix:end_ix] for arg in args])
        out = process_batch.scan(sequences = [start_ixs, start_ixs+batch_size])
        return out.reshape((-1, )+tuple(out.shape[i] for i in xrange(2, out.ndim)), ndim=out.ndim-1)
    return batch_function


@symbolic
def on_first_pass(first, after):
    """
    Return some value on the first past, and another after that.  This may be useful, for example, when a variable's
    value depends on shared variables whose shapes have not yet been initialized.

    :param first: Value to return on first pass
    :param after: Value to return after that.
    :return: first, if called on the first pass, otherwise after.
    """
    first_switch = theano.shared(1, 'initializing_switch')
    add_update(first_switch, 0)
    return ifelse(first_switch, first, after)


class ReshapingVariable(TensorVariable):

    def __add__(self, other):
        # if np.isscalar(other):

        return ifelse(self.size>0, tt.add(self, other), other+self.initial_value)

    def __sub__(self, other):
        return ifelse(self.size>0, tt.sub(self, other), other-self.initial_value)

    def __mul__(self, other):
        return ifelse(self.size>0, tt.mul(self, other), other*self.initial_value)

    # @classmethod
    # def create_reshaping(cls):



class ReshapingSharedVariable(TensorSharedVariable):
    """A shared variable with a dynamic shape."""

    def __add__(self, other):
        # if np.isscalar(other):

        return ifelse(self.size>0, tt.add(self, other), other+self.initial_value)

    def __sub__(self, other):
        return ifelse(self.size>0, tt.sub(self, other), other-self.initial_value)

    def __mul__(self, other):
        return ifelse(self.size>0, tt.mul(self, other), other*self.initial_value)

    # def __add__(self, other):
    #     return ifelse(self.size>0, tt.add(self, other), other+self.initial_value)
    #
    # def __mul__(self, other):
    #     return ifelse(self.size>0, tt.mul(self, other), other*self.initial_value)



def shared_of_type(ndim, value=0., dtype=theano.config.floatX, **kwargs):
    """
    Return a Shared Variable with dynamic shape. e.g.
        accumulator = shared_like(x)
        new_accum_val = accumulator+x
    :param ndim: Number of dimensions
    :param dtype: Data type
    :param kwargs: Passed to theano.shared
    :return: A ReshapingSharedVariable
    """
    out = theano.shared(np.zeros((0, )*ndim, dtype=dtype), **kwargs)
    out.__class__ = ReshapingSharedVariable
    out.initial_value = value
    return out


def shared_like(x, value=0., **kwargs):
    """
    Return a Shared Variable with dynamic shape. e.g.
        accumulator = shared_like(x)
        new_accum_val = accumulator+x
    :param x: A symbolic variable
    :param value: The initial value for this variable
    :param kwargs: Other args to pass to theano.shared
    :return: A ReshapingSharedVariabe
    """
    return shared_of_type(ndim=x.ndim, dtype=x.dtype, value=value, **kwargs)
