from collections import namedtuple
import theano
import numpy as np
import theano.tensor as tt
from plato.interfaces.decorators import symbolic_stateless, symbolic_updater
from theano.tensor.shared_randomstreams import RandomStreams
from utils.tools.mymath import bernoulli

__author__ = 'peter'

SamplingRegressor = namedtuple('SamplingRegressor', ('update', 'sample_posterior'))


def simple_binary_gibbs_regressor(n_dim_in, n_dim_out, sample_y = False, seed = None):
    """
    Returns the simplest form of a binary regressor.

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


def binary_gibbs_regressor(n_dim_in, n_dim_out, sample_y = False, seed = None):
    """
    Returns the simplest form of a binary regressor.

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


def herded_binary_gibbs_regressor(n_dim_in, n_dim_out, sample_y = False, seed = None):

    w = theano.shared(np.zeros((n_dim_in, n_dim_out), dtype = 'int'), name = 'w')
    phi = theano.shared(np.zeros((n_dim_in, n_dim_out), dtype = 'float'), name = 'phi')

    rng = RandomStreams(seed)
    alpha = theano.shared(np.array(0))

    @symbolic_updater
    def update(x, y):
        w_0 = tt.set_subtensor(w[alpha], 0)  # (n_dim_in, n_dim_out)
        w_1 = tt.set_subtensor(w[alpha], 1)  # (n_dim_in, n_dim_out)
        z_0 = tt.nnet.sigmoid(x.dot(w_0))  # (n_samples, n_dim_out)
        z_1 = tt.nnet.sigmoid(x.dot(w_1))  # (n_samples, n_dim_out)
        log_likelihood_ratio = tt.sum(tt.log(bernoulli(y, z_1))-tt.log(bernoulli(y, z_0)), axis = 0)  # (n_dim_out, )
        p_wa = tt.nnet.sigmoid(log_likelihood_ratio)  # (n_dim_out, )

        # Now, the herding part... here're the 3 lines from the minipaper
        phi_alpha = phi[alpha] + p_wa
        w_sample = phi_alpha > 0.5
        new_phi_alpha = phi_alpha - w_sample

        new_phi = tt.set_subtensor(phi[alpha], new_phi_alpha)
        w_new = tt.set_subtensor(w[alpha], w_sample)  # (n_dim_in, n_dim_out)

        # showloc()
        return [(w, w_new), (phi, new_phi), (alpha, (alpha+1) % n_dim_in)]

    @symbolic_stateless
    def sample_posterior(x):
        p_y = tt.nnet.sigmoid(x.dot(w))
        return rng.binomial(p = p_y) if sample_y else p_y

    return SamplingRegressor(update=update, sample_posterior=sample_posterior)