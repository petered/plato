from general.numpy_helpers import get_rng
import numpy as np
from plato.core import symbolic, symbolic_updater, symbolic_simple, tdbprint
from plato.interfaces.helpers import get_theano_rng
from plato.tools.mlp.mlp import Layer
from plato.tools.optimization.optimizers import AdaMax
import theano.tensor as tt
__author__ = 'peter'


class GaussianVariationalAutoencoder(object):
    """
    A Variational Autoencoder with gaussian latent space and either gaussian or binary input space.
    As described in:
    Kingma D, Welling M.  Auto-Encoding Variational Bayes
    http://arxiv.org/pdf/1312.6114v10.pdf

    This is a less configurable but easier-to-read version of of the one in variational_autoencoder.py
    """

    def __init__(self, x_dim, z_dim, encoder_hidden_sizes = [100], decoder_hidden_sizes = [100],
                 hidden_activation = 'tanh', w_init_mag = 0.01, binary_data = False,
                 optimizer = AdaMax(alpha = 0.01), rng = None, gaussian_min_var = None):
        """
        :param x_dim: Dimensionsality of the data
        :param z_dim: Dimensionalality of the latent space
        :param encoder_hidden_sizes: A list of sizes of each hidden layer in the encoder (from X to Z)
        :param decoder_hidden_sizes: A list of sizes of each hidden layer in the dencoder (from Z to X)
        :param hidden_activation: Activation function for all hidden layers
        :param w_init_mag: Magnitude of initial weights
        :param binary_data: Chose if data is binary.  You can also use this if data is bound in [0, 1] - then we can think
            of it as being the expected value.
        :param optimizer: An IGradientOptimizer object for doing parameter updates
            ... see plato.tools.optimization.optimizers
        :param rng: A random number generator or random seed.
        """
        np_rng = get_rng(rng)

        encoder_layer_sizes = [x_dim]+encoder_hidden_sizes
        self.encoder_hidden_layers = [Layer(w_init_mag*np_rng.randn(n_in, n_out), nonlinearity=hidden_activation)
                                      for n_in, n_out in zip(encoder_layer_sizes[:-1], encoder_layer_sizes[1:])]
        self.encoder_mean_layer = Layer(w_init_mag*np_rng.randn(encoder_layer_sizes[-1], z_dim), nonlinearity='linear')
        self.encoder_log_var_layer = Layer(w_init_mag*np_rng.randn(encoder_layer_sizes[-1], z_dim), nonlinearity='linear')

        decoder_layer_sizes = [z_dim] + decoder_hidden_sizes
        self.decoder_hidden_layers = [Layer(w_init_mag*np_rng.randn(n_in, n_out), nonlinearity=hidden_activation)
                                      for n_in, n_out in zip(decoder_layer_sizes[:-1], decoder_layer_sizes[1:])]
        if binary_data:
            self.decoder_mean_layer = Layer(w_init_mag*np_rng.randn(decoder_layer_sizes[-1], x_dim), nonlinearity='sigm')
        else:
            self.decoder_mean_layer = Layer(w_init_mag*np_rng.randn(decoder_layer_sizes[-1], x_dim), nonlinearity='linear')
            self.decoder_log_var_layer = Layer(w_init_mag*np_rng.randn(decoder_layer_sizes[-1], x_dim), nonlinearity='linear')

        self.rng = get_theano_rng(np_rng)
        self.binary_data = binary_data
        self.x_size = x_dim
        self.z_size = z_dim
        self.optimizer = optimizer
        self.gaussian_min_var = gaussian_min_var

    @symbolic_updater
    def train(self, x_samples):
        """
        :param x_samples: An (n_samples, n_dims) array of inputs
        :return: A list of training updates
        """
        lower_bound = self.compute_lower_bound(x_samples)
        updates = self.optimizer(cost = -lower_bound, parameters = self.parameters)
        return updates

    @symbolic_simple
    def compute_lower_bound(self, x_samples):
        """
        :param x_samples: An (n_samples, x_dim) array
        :return: A scalar - the mean lower bound on the log-probability of the data
        """
        z_mean, z_log_var = self.encode(x_samples)  # (n_samples, z_size), (n_samples, z_size)
        z_sample = self._sample_z_given_z_dist(z_mean=z_mean, z_log_var=z_log_var)
        z_sigma_sq = tt.exp(z_log_var)
        kl_divergence = -.5*tt.sum(1+tt.log(z_sigma_sq) - z_mean**2 - z_sigma_sq, axis = 1)
        if self.binary_data:
            x_mean = self.decode(z_sample)
            log_prop_data = tt.sum(x_samples*tt.log(x_mean) + (1-x_samples)*tt.log(1-x_mean), axis = 1)
        else:
            x_mean, x_log_var = self.decode(z_sample)
            x_sigma_sq = tt.exp(x_log_var)
            if self.gaussian_min_var is not None:
                x_sigma_sq = tt.maximum(self.gaussian_min_var, x_sigma_sq)
            log_prop_data = tt.sum(-0.5*tt.log(2*np.pi*x_sigma_sq)-((x_samples-x_mean)**2)/(2*x_sigma_sq), axis = 1)
        lower_bound = -kl_divergence + log_prop_data
        return lower_bound.mean()

    @symbolic
    def encode(self, x):
        """
        :param x: An (n_samples, x_dim) array
        :return: z_mean, z_log_var, each of which is an (n_samples, n_dims) array.
        """
        h=x
        for layer in self.encoder_hidden_layers:
            h = layer(h)
        z_means = self.encoder_mean_layer(h)
        z_log_vars = self.encoder_log_var_layer(h)
        return z_means, z_log_vars

    @symbolic
    def decode(self, z):
        """
        :param z: An (n_samples, n_z_dims) array of points in Z
        :return: Either:
            A (n_samples, n_x_dims) array of means (in the case of bernoulli-distributed X), OR
            Two arrays (means, log_vars), where each is (n_samples, n_x_dims) (in the case of Gaussian-distributed X)
        """
        h=z
        for layer in self.decoder_hidden_layers:
            h = layer(h)
        if self.binary_data:
            return self.decoder_mean_layer(h)
        else:
            x_means = self.decoder_mean_layer(h)
            x_log_vars = self.decoder_log_var_layer(h)
            return x_means, x_log_vars

    @symbolic_simple
    def sample(self, n_samples):
        """
        Draw samples from the model
        :param n_samples: The number of samples to draw
        :return: An (n_samples, x_dims) array of model samples.
        """
        z_samples = self.rng.normal(size = (n_samples, self.z_size))
        return self.sample_x_given_z(z_samples)

    @symbolic_simple
    def sample_x_given_z(self, z_samples):
        """
        Sample x given the distribution of the parameters
        :param params:
        :return:
        """
        if self.binary_data:
            x_mean = self.decode(z_samples)
            x_samples = self.rng.binomial(p=x_mean, size = x_mean.shape)
        else:
            x_mean, x_log_var = self.decode(z_samples)
            x_samples = x_mean + tt.sqrt(tt.exp(x_log_var)) * self.rng.normal(size = (z_samples.shape[0], self.x_size))
        return x_samples

    def _sample_z_given_z_dist(self, z_mean, z_log_var):
        epsilon = self.rng.normal(size = z_mean.shape)
        z_sample = epsilon * tt.sqrt(tt.exp(z_log_var)) + z_mean  # (n_samples, z_size).  Reparametrization trick!
        return z_sample

    @symbolic_simple
    def recon(self, x_samples):
        """
        Stochastic reconstruction of input vectors.
        """
        z_mean, z_log_var = self.encode(x_samples)  # (n_samples, z_size), (n_samples, z_size)
        z_samples = self._sample_z_given_z_dist(z_mean, z_log_var)
        x_recon_samples = self.sample_x_given_z(z_samples)
        return x_recon_samples

    @property
    def parameters(self):
        all_params = sum([l.parameters for l in self.encoder_hidden_layers], []) + self.encoder_mean_layer.parameters + self.encoder_log_var_layer.parameters \
            + sum([l.parameters for l in self.decoder_hidden_layers], []) + self.decoder_mean_layer.parameters \
            + (self.decoder_log_var_layer.parameters if not self.binary_data else [])
        return all_params
