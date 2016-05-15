import numpy as np
from plato.core import symbolic_simple, add_update, create_shared_variable
from plato.interfaces.interfaces import IParameterized
from plato.tools.common.basic import softmax
import theano
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


normalize= lambda x, axis = None: x/(x.sum(axis=axis, keepdims = True) + 1e-9)

normalize_safely= lambda x, axis = None, degree = 1: x/((x**degree).sum(axis=axis, keepdims = True) + 1)**(1./degree)


def softmax(x, axis=1):
    # Slightly more general than theano's softmax, in that it works on arbitrarily shaped arrays
    e_x = tt.exp(x - x.max(axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)



def get_named_activation_function(activation_name):
    fcn = {
        'softmax': softmax,
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
        'linear': lambda x: x,
        'leaky-relu-0.01': lambda x: tt.maximum(0.01*x, x),
        'maxout': lambda x: tt.max(x, axis=1),  # We expect (n_samples, n_maps, n_dims) data and flatten to (n_samples, n_dims)
        }[activation_name]
    return symbolic_simple(fcn)


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
