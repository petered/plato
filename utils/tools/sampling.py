import theano
import numpy as np
import theano.tensor as tt
from plato.interfaces.decorators import symbolic_stateless, symbolic_updater
from theano.tensor.shared_randomstreams import RandomStreams
from utils.tools.mymath import bernoulli

__author__ = 'peter'


class GibbsRegressor(object):

    def __init__(self, n_dim_in, n_dim_out, sample_y = False, n_alpha = 1, possible_ws = [0, 1], seed = None):
        self._w = theano.shared(np.zeros((n_dim_in, n_dim_out), dtype = 'int'), name = 'w')
        self._rng = RandomStreams(seed)
        if n_alpha == 'all':
            n_alpha = n_dim_in
        self._n_alpha = n_alpha
        self._alpha = theano.shared(np.arange(n_alpha))  # scalar
        self._sample_y = sample_y
        self._possible_ws = theano.shared(np.array(possible_ws), name = 'possible_ws')

    @staticmethod
    def compute_p_wa(w, x, y, alpha, possible_ws = np.array([0, 1])):
        """
        Compute the probability the weights at index alpha taking on
        value 1.

        """
        v_current = x.dot(w)  # (n_samples, n_dim_out)
        v_0 = v_current[None, :, :] - w[alpha, None, :]*x.T[alpha, :, None]  # (n_alpha, n_samples, n_dim_out)
        possible_vs = v_0[:, :, :, None] + possible_ws[None, None, None, :]*x.T[alpha, :, None, None]  # (n_alpha, n_samples, n_dim_out, n_possible_ws)
        all_zs = tt.nnet.sigmoid(possible_vs)  # (n_alpha, n_samples, n_dim_out, n_possible_ws)
        log_likelihoods = tt.sum(tt.log(bernoulli(y[None, :, :, None], all_zs)), axis = 1)  # (n_alpha, n_dim_out, n_possible_ws)
        # Question: Need to shift for stability here or will Theano take care of that?
        # Stupid theano didn't implement softmax very nicely so we have to do some reshaping.
        return tt.nnet.softmax(log_likelihoods.reshape([alpha.shape[0]*w.shape[1], possible_ws.shape[0]]))\
            .reshape([alpha.shape[0], w.shape[1], possible_ws.shape[0]])  # (n_alpha, n_dim_out, n_possible_ws)

    @symbolic_updater
    def update(self, x, y):
        p_wa = self.compute_p_wa(self._w, x, y, self._alpha, self._possible_ws)  # (n_alpha, n_dim_out, n_possible_ws)
        w_indices = sample_categorical(self._rng, p_wa)
        w_sample = self._possible_ws[w_indices]  # (n_alpha, n_dim_out)
        w_new = tt.set_subtensor(self._w[self._alpha], w_sample)  # (n_dim_in, n_dim_out)
        return [(self._w, w_new), (self._alpha, (self._alpha+self._n_alpha) % self._w.shape[0])]

    @symbolic_stateless
    def sample_posterior(self, x):
        p_y = tt.nnet.sigmoid(x.dot(self._w))
        return self._rng.binomial(p = p_y) if self._sample_y else p_y


class HerdedGibbsRegressor(GibbsRegressor):

    def __init__(self, n_dim_in, n_dim_out, possible_ws = (0, 1), **kwargs):
        GibbsRegressor.__init__(self, n_dim_in, n_dim_out, possible_ws=possible_ws, **kwargs)
        self._phi = theano.shared(np.zeros((n_dim_in, n_dim_out, len(possible_ws)), dtype = 'float'), name = 'phi')

    @symbolic_updater
    def update(self, x, y):
        p_wa = self.compute_p_wa(self._w, x, y, self._alpha, self._possible_ws)
        phi_alpha = self._phi[self._alpha] + p_wa  # (n_alpha, n_dim_out, n_possible_ws)

        k_chosen = tt.argmax(phi_alpha, axis = 2)  # (n_alpha, n_dim_out)
        selected_phi_indices = (tt.arange(self._alpha.shape[0])[:, None], tt.arange(y.shape[1])[None, :], k_chosen)
        new_phi_alpha = tt.set_subtensor(phi_alpha[selected_phi_indices], phi_alpha[selected_phi_indices]-1)  # (n_alpha, n_dim_out, n_possible_ws)
        w_sample = self._possible_ws[k_chosen]  # (n_alpha, n_dim_out)

        # w_sample = phi_alpha > 0.5  # TODOOOOOO: Updata this.
        # new_phi_alpha = phi_alpha - w_sample
        new_phi = tt.set_subtensor(self._phi[self._alpha], new_phi_alpha)  # (n_dim_in, n_dim_out, n_possible_ws)
        w_new = tt.set_subtensor(self._w[self._alpha], w_sample)  # (n_dim_in, n_dim_out)
        return [(self._w, w_new), (self._phi, new_phi), (self._alpha, (self._alpha+1) % self._w.shape[0])]


def sample_categorical(rng, p):
    """
    p is a n-d array, where the final dimension is a discrete distibution (does not need to be normalized).
    Sample from that distribution.
    This will return an array of shape p.shape[:-1] with values in range [0, p.shape[-1])
    """
    p = p/tt.sum(p, axis = -1)[(slice(None), )*(p.ndim-1)+(None, )]
    samples = rng.multinomial(n=1, pvals = p)
    indices = tt.argmax(samples, axis = -1)
    return indices


# SamplingRegressor = namedtuple('SamplingRegressor', ('update', 'sample_posterior'))
#
#
# @symbolic_stateless
# def compute_p_wa(w, x, y, alpha):
#     w_0 = tt.set_subtensor(w[alpha], 0)  # (n_dim_in, n_dim_out)
#     w_1 = tt.set_subtensor(w[alpha], 1)  # (n_dim_in, n_dim_out)
#     z_0 = tt.nnet.sigmoid(x.dot(w_0))  # (n_samples, n_dim_out)
#     z_1 = tt.nnet.sigmoid(x.dot(w_1))  # (n_samples, n_dim_out)
#     log_likelihood_ratio = tt.sum(tt.log(bernoulli(y, z_1))-tt.log(bernoulli(y, z_0)), axis = 0)  # (n_dim_out, )
#     p_wa = tt.nnet.sigmoid(log_likelihood_ratio)  # (n_dim_out, )
#     return p_wa
#
#
# def binary_gibbs_regressor(n_dim_in, n_dim_out, sample_y = False, seed = None):
#     """
#     Returns the simplest form of a binary regressor.
#
#     :param n_dim_in: Number of dimensions of the input
#     :param n_dim_out: Number of dimensions of the output
#     :param sample_y: Sample output from a bernoulli distribution (T) or return the probability (F)
#     :param seed: Seed for the random number generator.
#     :return: A SamplingRegressor object containing the functions for updating and sampling the posterior.
#     """
#
#     w = theano.shared(np.zeros((n_dim_in, n_dim_out), dtype = 'int'), name = 'w')
#     rng = RandomStreams(seed)
#     alpha = theano.shared(np.array(0))  # scalar
#
#     @symbolic_updater
#     def update(x, y):
#         p_wa = compute_p_wa(w, x, y, alpha)
#         w_sample = rng.binomial(p=p_wa)  # (n_dim_out, )
#         w_new = tt.set_subtensor(w[alpha], w_sample)  # (n_dim_in, n_dim_out)
#         return [(w, w_new), (alpha, (alpha+1) % n_dim_in)]
#
#     @symbolic_stateless
#     def sample_posterior(x):
#         p_y = tt.nnet.sigmoid(x.dot(w))
#         return rng.binomial(p = p_y) if sample_y else p_y
#
#     return SamplingRegressor(update=update, sample_posterior=sample_posterior)
#
#
#
#
# def herded_binary_gibbs_regressor(n_dim_in, n_dim_out, sample_y = False, seed = None):
#
#     w = theano.shared(np.zeros((n_dim_in, n_dim_out), dtype = 'int'), name = 'w')
#     phi = theano.shared(np.zeros((n_dim_in, n_dim_out), dtype = 'float'), name = 'phi')
#
#     rng = RandomStreams(seed)
#     alpha = theano.shared(np.array(0))
#
#     @symbolic_updater
#     def update(x, y):
#         p_wa = compute_p_wa(w, x, y, alpha)
#
#         # Now, the herding part... here're the 3 lines from the minipaper
#         phi_alpha = phi[alpha] + p_wa
#         w_sample = phi_alpha > 0.5
#         new_phi_alpha = phi_alpha - w_sample
#
#         new_phi = tt.set_subtensor(phi[alpha], new_phi_alpha)
#         w_new = tt.set_subtensor(w[alpha], w_sample)  # (n_dim_in, n_dim_out)
#
#         # showloc()
#         return [(w, w_new), (phi, new_phi), (alpha, (alpha+1) % n_dim_in)]
#
#     @symbolic_stateless
#     def sample_posterior(x):
#         p_y = tt.nnet.sigmoid(x.dot(w))
#         return rng.binomial(p = p_y) if sample_y else p_y
#
#     return SamplingRegressor(update=update, sample_posterior=sample_posterior)
#
#
#
# def simple_binary_gibbs_regressor(n_dim_in, n_dim_out, sample_y = False, seed = None):
#     """
#     Returns the simplest form of a binary regressor.  We use this as an example.  For a version with
#     more parameters, see binary_gibbs_regressor.
#
#     :param n_dim_in: Number of dimensions of the input
#     :param n_dim_out: Number of dimensions of the output
#     :param sample_y: Sample output from a bernoulli distribution (T) or return the probability (F)
#     :param seed: Seed for the random number generator.
#     :return: A SamplingRegressor object containing the functions for updating and sampling the posterior.
#     """
#
#     w = theano.shared(np.zeros((n_dim_in, n_dim_out), dtype = 'int'), name = 'w')
#     rng = RandomStreams(seed)
#     alpha = theano.shared(np.array(0))  # scalar
#
#     @symbolic_updater
#     def update(x, y):
#         w_0 = tt.set_subtensor(w[alpha], 0)  # (n_dim_in, n_dim_out)
#         w_1 = tt.set_subtensor(w[alpha], 1)  # (n_dim_in, n_dim_out)
#         z_0 = tt.nnet.sigmoid(x.dot(w_0))  # (n_samples, n_dim_out)
#         z_1 = tt.nnet.sigmoid(x.dot(w_1))  # (n_samples, n_dim_out)
#         log_likelihood_ratio = tt.sum(tt.log(bernoulli(y, z_1))-tt.log(bernoulli(y, z_0)), axis = 0)  # (n_dim_out, )
#         p_wa = tt.nnet.sigmoid(log_likelihood_ratio)  # (n_dim_out, )
#         w_sample = rng.binomial(p=p_wa)  # (n_dim_out, )
#         w_new = tt.set_subtensor(w[alpha], w_sample)  # (n_dim_in, n_dim_out)
#         return [(w, w_new), (alpha, (alpha+1) % n_dim_in)]
#
#     @symbolic_stateless
#     def sample_posterior(x):
#         p_y = tt.nnet.sigmoid(x.dot(w))
#         return rng.binomial(p = p_y) if sample_y else p_y
#
#     return SamplingRegressor(update=update, sample_posterior=sample_posterior)
