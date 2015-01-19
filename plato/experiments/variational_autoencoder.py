"""
An implementation of the Variational Autoencoder presented in the following paper:
http://arxiv.org/pdf/1312.6114v10.pdf
"""
import numpy as np
import theano
import theano.tensor as ts
__author__ = 'peter'


class VariationalAutoencoder(object):

    def __init__(self,
             posterior_dim,
             hidden_sizes,
             hidden_types,
             optimizer = None,
             seed = None,
             ):

        self._encoder = FeedForwardNet(hidden_sizes, hidden_types)
        self._decoder = FeedForwardNet(hidden_sizes, hidden_types)

        self._rng = np.random.RandomState(seed)

    def infer_posterior(self, x):
        """
        Return the posterior distribution given the visible data
        """
        mu_z, v_z = self._forward_network(x)

        cov_z = theano

        return Gaussian(mu_z, cov_z)

    def infer_conditional(self, z):
        """

        """
        mu_x, cov_x = self._decoder(z)
        return Gaussian(mu_x, cov_x)

    def learn(self, x):

        z = self.infer_posterior(x)

        noise_sample = self._rng.randn(len(x), self._posterior_dim)

        z_sample = ts.mean(z.mean[None] + z.cov * noise_sample, axis = 0)




