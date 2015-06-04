from abc import abstractproperty, abstractmethod
from general.should_be_builtins import bad_value
import numpy as np
from plato.interfaces.decorators import symbolic_updater, symbolic_stateless
from plato.interfaces.helpers import get_theano_rng
from plato.interfaces.interfaces import IParameterized
from plato.tools.linking import Chain, Branch
from plato.tools.networks import FullyConnectedBridge, Layer
from plato.tools.optimizers import AdaMax
import theano.tensor as tt
__author__ = 'peter'

"""
Implementation of a variational autoencoder.

Paper: http://arxiv.org/pdf/1312.6114v10.pdf
Prior code example: https://github.com/y0ast/Variational-Autoencoder/blob/master/Theano/VariationalAutoencoder.py
"""


class VariationalAutoencoder(object):
    """
    A Variational Autoencoder, as described in
    Kingma D, Welling M.  Auto-Encoding Variational Bayes
    http://arxiv.org/pdf/1312.6114v10.pdf
    """

    def __init__(self, pq_pair, optimizer = AdaMax(alpha = 0.01), rng = None):
        """
        :param pq_pair: An IVeriationalPair object
        :param optimizer: An IGradientOptimizer object
        :param rng: A random number generator, or seed.
        """
        self.rng = get_theano_rng(rng)
        self.pq_pair = pq_pair
        self.optimizer = optimizer

    @symbolic_updater
    def train(self, x_samples):
        z_dist = self.pq_pair.p_z_given_x(x_samples)
        z_samples = z_dist.sample(1, self.rng)[0]  # Just one sample per data point.  Shape (minibatch_size, n_dims)
        x_dist = self.pq_pair.p_x_given_z(z_samples)
        lower_bound = -z_dist.kl_divergence(self.pq_pair.prior) + x_dist.log_prob(x_samples) # (minibatch_size, )
        updates = self.optimizer(cost = -lower_bound.mean(), parameters = self.parameters)
        return updates

    @symbolic_stateless
    def sample(self, n_samples):
        z_samples = self.pq_pair.prior.sample(n_samples, self.rng)
        return self.sample_x_given_z(z_samples)

    @symbolic_stateless
    def sample_x_given_z(self, z_samples):
        x_dist = self.pq_pair.p_x_given_z(z_samples)
        x_samples = x_dist.sample(1, self.rng)[0]
        return x_samples

    @symbolic_stateless
    def sample_z_given_x(self, x_samples):
        z_dist = self.pq_pair.p_z_given_x(x_samples)
        z_samples = z_dist.sample(1, self.rng)[0]  # Just one sample per data point.  Shape (minibatch_size, n_dims)
        return z_samples

    @property
    def parameters(self):
        return self.pq_pair.parameters


class IVariationalPair(object):
    """
    A model defining the distributions p(X|Z), p(Z), and P(Z|X) should fulfill this interace.
    """

    @abstractproperty
    def prior(self):
        """ The prior - an IDistribution object """

    @abstractproperty
    def n_observed_dim(self):
        """ Number of observed dimensions """

    @abstractproperty
    def n_latent_dim(self):
        """ Number of latent dimensions """

    @abstractmethod
    def p_z_given_x(self, x):
        """
        Given a batch of samples x, return an IDistribution object defining the distribution over Z
        """

    @abstractmethod
    def p_x_given_z(self, z):
        """
        Given a batch of samples z, return an IDistribution object defining the distribution over X
        """

    @abstractproperty
    def parameters(self):
        pass


class EncoderDecoderNetworks(IVariationalPair):
    """
    An encoder/decoder pair that uses neural networks to encode the distributions p(Z|X) and p(X|Z).
    """

    def __init__(self, x_dim, z_dim, encoder_hidden_sizes = [100], decoder_hidden_sizes = [100],
                 hidden_activation = 'tanh', w_init = lambda n_in, n_out: 0.1*np.random.randn(n_in, n_out),
                 x_distribution = 'gaussian', z_distribution = 'gaussian'
                 ):

        self._n_observed_dim = x_dim
        self._n_latent_dim = z_dim
        self.p_net = DistributionMLP(input_size = z_dim, hidden_sizes = encoder_hidden_sizes,
            output_size=x_dim, hidden_activation=hidden_activation, w_init=w_init, distribution = x_distribution)
        self.q_net = DistributionMLP(input_size = x_dim, hidden_sizes = decoder_hidden_sizes,
            output_size=z_dim, hidden_activation=hidden_activation, w_init=w_init, distribution=z_distribution)
        self._prior = \
            StandardUniformDistribution(z_dim) if z_distribution == 'bernoulli' else \
            StandardNormalDistribution(z_dim) if z_distribution == 'gaussian' else \
            bad_value(z_distribution)

    @property
    def prior(self):
        return self._prior

    @property
    def n_observed_dim(self):
        return self._n_observed_dim

    @property
    def n_latent_dim(self):
        return self._n_latent_dim

    def p_z_given_x(self, x):
        return self.q_net(x)

    def p_x_given_z(self, z):
        return self.p_net(z)

    @property
    def parameters(self):
        return self.p_net.parameters + self.q_net.parameters


class DistributionMLP(IParameterized):
    """
    A Multi-Layer Perceptron that outputs a distribution.
    """

    def __init__(self, input_size, hidden_sizes, output_size, distribution = 'gaussian', hidden_activation = 'sig', w_init = lambda n_in, n_out: 0.01*np.random.randn(n_in, n_out)):
        """
        :param input_size: The dimensionality of the input
        :param hidden_sizes: A list indicating the sizes of each hidden layer.
        :param output_size: The dimensionality of the output
        :param distribution: The form of the output distribution (currently 'gaussian' or 'bernoulli')
        :param hidden_activation: A string indicating the type of each hidden layer.
            {'sig', 'tanh', 'rect-lin', 'lin', 'softmax'}
        :param w_init: A function which, given input dims, output dims, returns an initial weight matrix
        """

        all_layer_sizes = [input_size]+hidden_sizes

        all_layer_activations = [hidden_activation] * len(hidden_sizes)

        processing_chain = sum([[
             FullyConnectedBridge(w = w_init(pre_size, post_size)),
             Layer(activation_fcn)
             ] for (pre_size, post_size), activation_fcn in zip(zip(all_layer_sizes[:-1], all_layer_sizes[1:]), all_layer_activations)
             ], [])

        distribution_function = \
            Branch(
                 FullyConnectedBridge(w = w_init(all_layer_sizes[-1], output_size)),
                 FullyConnectedBridge(w_init(all_layer_sizes[-1], output_size))) \
                 if distribution == 'gaussian' else \
            Chain(FullyConnectedBridge(w = w_init(all_layer_sizes[-1], output_size)), Layer('sig')) \
                 if distribution=='bernoulli' else \
            bad_value(distribution)

        self.distribution = distribution
        self.chain = Chain(*processing_chain+[distribution_function])

    def __call__(self, x):

        if self.distribution == 'gaussian':
            (mu, log_sigma), _ = self.chain(x)
            dist = MultipleDiagonalGaussianDistribution(mu, sigma_sq = tt.exp(log_sigma)**2)

        elif self.distribution == 'bernoulli':
            (p, ), _ = self.chain(x)
            dist = MultipleBernoulliDistribution(p)
        return dist

    @property
    def parameters(self):
        return self.chain.parameters


class IDistribution(object):
    """
    An object representing a probability distribution.
    """

    def sample(self, n, rng):
        """
        :param rng: Draw n samples from this distribution.
        :return: A (n, n_dims) array of samples
        """

    def kl_divergence(self, other):
        """
        :param other: Another IDistribution, or maybe a string identifying a special case - e.g. 'standard-normal'
        :return:
        """
        raise NotImplementedError()

    def log_prob(self, point):
        """
        Compute the log probability of a point given the distribution
        :param point:
        :return:
        """
        raise NotImplementedError()


class StandardNormalDistribution(IDistribution):

    def __init__(self, n_dims):
        self.n_dims = n_dims

    def sample(self, n, rng):
        return rng.normal(size = (n, self.n_dims))


class MultipleDiagonalGaussianDistribution(IDistribution):
    """
    A collection of diagonal gaussian distributions
    """

    def __init__(self, mu, sigma_sq):
        """
        :param mu: An (n_samples, n_dims) vector of means
        :param sigma_sq: An (n_samples, n_dims) vector of variances
        """
        self.mu = mu
        self.sigma_sq = sigma_sq

    def sample(self, n, rng):
        mu_shape = self.mu.tag.test_value.shape
        return rng.normal(size = (n, )+mu_shape) * tt.sqrt(self.sigma_sq) + self.mu

    def kl_divergence(self, other):

        if isinstance(other, StandardNormalDistribution):
            return -.5*tt.sum(1+tt.log(self.sigma_sq) - self.mu**2 - self.sigma_sq, axis = 1)
        else:
            raise Exception("Don't know how to compute KL-divergence to %s" % (other, ))

    def log_prob(self, x):
        """
        :param x: Data is a (n_samples, n_dims) array of data
        :return: A length n_dims vector containing the log-probability of each data point given this distribution.
        """
        return tt.sum(-0.5*tt.log(2*np.pi*self.sigma_sq)-0.5*(x-self.mu)**2/self.sigma_sq, axis = 1)


class StandardUniformDistribution(IDistribution):

    def __init__(self, n_dims):
        self.n_dims = n_dims

    def sample(self, n, rng):
        return rng.uniform(size = (n, self.n_dims))


class MultipleBernoulliDistribution(IDistribution):

    def __init__(self, means):
        self.means = means

    def sample(self, n, rng):
        shape = self.means.tag.test_value.shape
        return self.means > rng.uniform(size = (n, )+shape)

    def log_prob(self, x):
        return tt.sum(x*tt.log(self.means) + (1-x)*tt.log(1-self.means), axis = 1)

    def kl_divergence(self, other):
        if isinstance(other, StandardUniformDistribution):
            return tt.sum(self.means*(tt.log(self.means) - tt.log(0.5)), axis = 1)
        else:
            raise NotImplementedError()
