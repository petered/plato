import numpy as np
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
            assert p is None, 'Only support uniform distributions right now'
            assert isinstance(size, int), 'Only support vectors right now'
            assert options.ndim == 1, 'Only suport vector values'
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