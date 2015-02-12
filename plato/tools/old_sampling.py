from collections import namedtuple
from plato.interfaces.decorators import symbolic_stateless, symbolic_updater
import theano
from theano.tensor.shared_randomstreams import RandomStreams
import theano.tensor as tt
import numpy as np
from utils.tools.mymath import bernoulli

__author__ = 'peter'

"""
Here we put correct but old samplers.  If they're here, they've been replaced by their more
general cousins, but we keep them around because they're unintimidating and can be used to
test the correctness of the general samplers in the special case.
"""

SamplingRegressor = namedtuple('SamplingRegressor', ('update', 'sample_posterior'))


def simple_binary_gibbs_regressor(n_dim_in, n_dim_out, sample_y = False, seed = None):
    """
    Returns the simplest form of a binary regressor.  We use this as an example.  For a version with
    more parameters, see binary_gibbs_regressor.

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
    def update(x, y):
        w_0 = tt.set_subtensor(w[alpha], 0)  # (n_dim_in, n_dim_out)
        w_1 = tt.set_subtensor(w[alpha], 1)  # (n_dim_in, n_dim_out)
        z_0 = tt.nnet.sigmoid(x.dot(w_0))  # (n_samples, n_dim_out)
        z_1 = tt.nnet.sigmoid(x.dot(w_1))  # (n_samples, n_dim_out)
        log_likelihood_ratio = tt.sum(tt.log(bernoulli(y, z_1))-tt.log(bernoulli(y, z_0)), axis = 0)  # (n_dim_out, )
        p_wa = tt.nnet.sigmoid(log_likelihood_ratio)  # (n_dim_out, )
        w_sample = rng.binomial(p=p_wa)  # (n_dim_out, )
        w_new = tt.set_subtensor(w[alpha], w_sample)  # (n_dim_in, n_dim_out)
        return [(w, w_new), (alpha, (alpha+1) % n_dim_in)]

    @symbolic_stateless
    def sample_posterior(x):
        p_y = tt.nnet.sigmoid(x.dot(w))
        return rng.binomial(p = p_y) if sample_y else p_y

    return SamplingRegressor(update=update, sample_posterior=sample_posterior)


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
    def update(self, x, y):
        p_wa = self.compute_p_wa(self._w, x, y, self._alpha, self._possible_ws)
        w_sample = self._rng.binomial(p=p_wa)  # (n_dim_out, )
        w_new = tt.set_subtensor(self._w[self._alpha], w_sample)  # (n_dim_in, n_dim_out)
        return [(self._w, w_new), (self._alpha, (self._alpha+self._n_alpha) % self._w.shape[0])]

    @symbolic_stateless
    def sample_posterior(self, x):
        p_y = tt.nnet.sigmoid(x.dot(self._w))
        return self._rng.binomial(p = p_y) if self._sample_y else p_y
