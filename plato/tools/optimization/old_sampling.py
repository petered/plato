from collections import namedtuple
from plato.core import add_update
from plato.interfaces.decorators import symbolic_simple, symbolic_updater
import theano
from theano.tensor.shared_randomstreams import RandomStreams
import theano.tensor as tt
import numpy as np

__author__ = 'peter'

"""
Here we put correct but old samplers.  If they're here, they've been replaced by their more
general cousins, but we keep them around because they're unintimidating and can be used to
test the correctness of the general samplers in the special case.
"""


bernoulli = lambda k, p: (p**k)*((1-p)**(1-k))  # Maybe a not the fastest way to do it but whatevs

def simple_binary_gibbs_regressor(n_dim_in, n_dim_out, sample_y = False, seed = None):
    """
    Returns the simplest form of a binary regressor.  We use this as an example.  For a more general version,
    see class BinaryGibbsRegressor.

    :param n_dim_in: Number of dimensions of the input
    :param n_dim_out: Number of dimensions of the output
    :param sample_y: Sample output from a bernoulli distribution (T) or return the probability (F)
    :param seed: Seed for the random number generator.
    :return: A SamplingRegressor object containing the functions for updating and sampling the posterior.
    """

    w = theano.shared(np.zeros((n_dim_in, n_dim_out), dtype = 'int'), name = 'w')
    rng = RandomStreams(seed)
    alpha = theano.shared(np.array(0))  # scalar

    @symbolic_updater
    def train(x, y):
        w_0 = tt.set_subtensor(w[alpha], 0)  # (n_dim_in, n_dim_out)
        w_1 = tt.set_subtensor(w[alpha], 1)  # (n_dim_in, n_dim_out)
        z_0 = tt.nnet.sigmoid(x.dot(w_0))  # (n_samples, n_dim_out)
        z_1 = tt.nnet.sigmoid(x.dot(w_1))  # (n_samples, n_dim_out)
        log_likelihood_ratio = tt.sum(tt.log(bernoulli(y, z_1))-tt.log(bernoulli(y, z_0)), axis = 0)  # (n_dim_out, )
        p_wa = tt.nnet.sigmoid(log_likelihood_ratio)  # (n_dim_out, )
        w_sample = rng.binomial(p=p_wa)  # (n_dim_out, )
        w_new = tt.set_subtensor(w[alpha], w_sample)  # (n_dim_in, n_dim_out)
        add_update(w, w_new)
        add_update(alpha, (alpha+1) % n_dim_in)

    @symbolic_simple
    def predict(x):
        p_y = tt.nnet.sigmoid(x.dot(w))
        return rng.binomial(p = p_y) if sample_y else p_y

    return SamplingRegressor(train=train, predict=predict)


SamplingRegressor = namedtuple('SamplingRegressor', ('train', 'predict'))


@symbolic_simple
def compute_p_wa(w, x, y, alpha):
    w_0 = tt.set_subtensor(w[alpha], 0)  # (n_dim_in, n_dim_out)
    w_1 = tt.set_subtensor(w[alpha], 1)  # (n_dim_in, n_dim_out)
    z_0 = tt.nnet.sigmoid(x.dot(w_0))  # (n_samples, n_dim_out)
    z_1 = tt.nnet.sigmoid(x.dot(w_1))  # (n_samples, n_dim_out)
    log_likelihood_ratio = tt.sum(tt.log(bernoulli(y, z_1))-tt.log(bernoulli(y, z_0)), axis = 0)  # (n_dim_out, )
    p_wa = tt.nnet.sigmoid(log_likelihood_ratio)  # (n_dim_out, )
    return p_wa


def simple_herded_binary_gibbs_regressor(n_dim_in, n_dim_out, sample_y = False, seed = None):

    w = theano.shared(np.zeros((n_dim_in, n_dim_out), dtype = 'int'), name = 'w')
    phi = theano.shared(np.zeros((n_dim_in, n_dim_out), dtype = 'float'), name = 'phi')

    rng = RandomStreams(seed)
    alpha = theano.shared(np.array(0))

    @symbolic_updater
    def train(x, y):
        p_wa = compute_p_wa(w, x, y, alpha)

        # Now, the herding part... here're the 3 lines from the minipaper
        phi_alpha = phi[alpha] + p_wa
        w_sample = phi_alpha > 0.5
        new_phi_alpha = phi_alpha - w_sample
        add_update(w, tt.set_subtensor(w[alpha], w_sample))
        add_update(phi, tt.set_subtensor(phi[alpha], new_phi_alpha))
        add_update(alpha, (alpha+1) % n_dim_in)

    @symbolic_simple
    def predict(x):
        p_y = tt.nnet.sigmoid(x.dot(w))
        return rng.binomial(p = p_y) if sample_y else p_y

    return SamplingRegressor(train=train, predict=predict)


class OldGibbsRegressor(object):

    def __init__(self, n_dim_in, n_dim_out, sample_y = False, n_alpha = 1, seed = None):
        self._w = theano.shared(np.zeros((n_dim_in, n_dim_out), dtype = 'int'), name = 'w')
        self._rng = RandomStreams(seed)
        if n_alpha == 'all':
            n_alpha = n_dim_in
        self._n_alpha = n_alpha
        self._alpha = theano.shared(np.arange(n_alpha))  # scalar
        self._sample_y = sample_y

    @staticmethod
    def compute_p_wa(w, x, y, alpha):
        """
        Compute the probability the weights at index alpha taking on
        value 1.
        """
        w_0 = tt.set_subtensor(w[alpha], 0)  # (n_dim_in, n_dim_out)
        w_1 = tt.set_subtensor(w[alpha], 1)  # (n_dim_in, n_dim_out)
        z_0 = tt.nnet.sigmoid(x.dot(w_0))  # (n_samples, n_dim_out)
        z_1 = tt.nnet.sigmoid(x.dot(w_1))  # (n_samples, n_dim_out)
        log_likelihood_ratio = tt.sum(tt.log(bernoulli(y, z_1))-tt.log(bernoulli(y, z_0)), axis = 0)  # (n_dim_out, )
        p_wa = tt.nnet.sigmoid(log_likelihood_ratio)  # (n_dim_out, )
        return p_wa

    @symbolic_updater
    def train(self, x, y):
        p_wa = self.compute_p_wa(self._w, x, y, self._alpha)
        w_sample = self._rng.binomial(p=p_wa)  # (n_dim_out, )
        w_new = tt.set_subtensor(self._w[self._alpha], w_sample)  # (n_dim_in, n_dim_out)
        add_update(self._w, w_new)
        add_update(self._alpha, (self._alpha+self._n_alpha) % self._w.shape[0])

    @symbolic_simple
    def predict(self, x):
        p_y = tt.nnet.sigmoid(x.dot(self._w))
        return self._rng.binomial(p = p_y) if self._sample_y else p_y
