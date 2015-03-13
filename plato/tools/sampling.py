import theano
import numpy as np
import theano.tensor as tt
from plato.interfaces.decorators import symbolic_stateless, symbolic_updater
from theano.tensor.shared_randomstreams import RandomStreams
from utils.predictors.i_predictor import IPredictor
from utils.tools.mymath import bernoulli

__author__ = 'peter'


class GibbsRegressor(object):

    def __init__(self, n_dim_in, n_dim_out, sample_y = False, n_alpha = 1, possible_ws = [0, 1],
            alpha_update_policy = 'sequential', seed = None):
        self._w = theano.shared(np.zeros((n_dim_in, n_dim_out), dtype = theano.config.floatX), name = 'w')
        self._rng = RandomStreams(seed)
        if n_alpha == 'all':
            n_alpha = n_dim_in
        self._n_alpha = n_alpha
        self._alpha = theano.shared(np.arange(n_alpha))  # scalar
        self._sample_y = sample_y
        self._possible_ws = theano.shared(np.array(possible_ws), name = 'possible_ws')
        assert alpha_update_policy in ('sequential', 'random')
        self._alpha_update_policy = alpha_update_policy

    def _get_alpha_update(self):
        new_alpha = (self._alpha+self._n_alpha) % self._w.shape[0] \
            if self._alpha_update_policy == 'sequential' else \
            self._rng.choice(a=self._w.shape[0], size = (self._n_alpha, ), replace = False).reshape([-1])  # Reshape is for some reason necessary when n_alpha=1
        return self._alpha, new_alpha

    @staticmethod
    def compute_p_wa(w, x, y, alpha, possible_ws = np.array([0, 1])):
        """
        Compute the probability the weights at index alpha taking on each of the values in possible_ws
        """
        assert x.tag.test_value.ndim == y.tag.test_value.ndim == 2
        assert x.tag.test_value.shape[0] == y.tag.test_value.shape[0]
        assert w.get_value().shape[1] == y.tag.test_value.shape[1]
        v_current = x.dot(w)  # (n_samples, n_dim_out)
        v_0 = v_current[None, :, :] - w[alpha, None, :]*x.T[alpha, :, None]  # (n_alpha, n_samples, n_dim_out)
        possible_vs = v_0[:, :, :, None] + possible_ws[None, None, None, :]*x.T[alpha, :, None, None]  # (n_alpha, n_samples, n_dim_out, n_possible_ws)
        all_zs = tt.nnet.sigmoid(possible_vs)  # (n_alpha, n_samples, n_dim_out, n_possible_ws)
        log_likelihoods = tt.sum(tt.log(bernoulli(y[None, :, :, None], all_zs[:, :, :, :])), axis = 1)  # (n_alpha, n_dim_out, n_possible_ws)
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
        return [(self._w, w_new), self._get_alpha_update()]

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
        new_phi = tt.set_subtensor(self._phi[self._alpha], new_phi_alpha)  # (n_dim_in, n_dim_out, n_possible_ws)
        w_new = tt.set_subtensor(self._w[self._alpha], w_sample)  # (n_dim_in, n_dim_out)
        return [(self._w, w_new), (self._phi, new_phi), self._get_alpha_update()]


def sample_categorical(rng, p, axis = -1):
    """
    p is a n-d array, where the final dimension is a discrete distibution (does not need to be normalized).
    Sample from that distribution.
    This will return an array of shape p.shape[:-1] with values in range [0, p.shape[-1])
    """
    assert axis==-1, 'Currenly you can only sample along the last axis.'
    p = p/tt.sum(p, axis = -1, keepdims=True)
    samples = rng.multinomial(n=1, pvals = p)
    indices = tt.argmax(samples, axis = -1)  # Argmax is just a way to find the location of the only element that is 1.
    return indices


class SamplingPredictor(IPredictor):
    """
    Deprecated.

    TODO: Replace with plato.tools.online_prediction.CompiledSymbolicPredictor
    """

    def __init__(self, sampler, mode = 'test_and_run'):
        self.train_function = sampler.update.compile(mode=mode)
        self.predict_function = sampler.sample_posterior.compile(mode=mode)

    def train(self, input_data, target_data):
        self.train_function(input_data, target_data)

    def predict(self, input_data):
        return self.predict_function(input_data)
