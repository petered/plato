from plato.core import symbolic
from plato.interfaces.helpers import get_theano_rng
from plato.tools.misc.tdb_plotting import tdbplot
import theano.tensor as tt
__author__ = 'peter'


class GenerativeAdversarialNetwork(object):

    def __init__(self, generator, discriminator, noise_dim, optimizer, rng=None):
        """
        :param generator: Takes A (n_samples, noise_dim, ) array of gaussian random noise, and creates a
            (n_samples, sample_dim) array of sample points.
        :param discriminator: Takes a (n_samples, sample_dim) array of sample points, outputs a scalar probability of
            the sample being from the data, as opposed to the generator.
        :return:
        """
        self.generator = generator
        self.discriminator = discriminator
        self.noise_dim = noise_dim
        self.optimizer = optimizer
        self.rng = get_theano_rng(rng)

    @symbolic
    def train_discriminator(self, data):
        """
        :param data: An (n_samples, n_dims) array of data
        :return:
        """
        counterfeit_samples = self.generator(self.rng.normal(size=(data.shape[0], self.noise_dim)))
        real_sample_guesses = self.discriminator(data)
        counterfeit_sample_guesses = self.discriminator(counterfeit_samples)

        # tdbplot(real_sample_guesses.mean(), 'Real prob')
        # tdbplot(counterfeit_sample_guesses.mean(), 'Fake prob')

        self.optimizer(cost = -(tt.log(real_sample_guesses) + tt.log(1-counterfeit_sample_guesses)).mean(), parameters = self.discriminator.parameters)

    @symbolic
    def train_generator(self, n_samples):
        counterfeit_sample_guesses = self.discriminator(self.generator(self.rng.normal(size=(n_samples, self.noise_dim))))
        self.optimizer(cost = tt.log(1-counterfeit_sample_guesses).mean(), parameters = self.generator.parameters)
        # self.optimizer(cost = -tt.log(counterfeit_sample_guesses).mean(), parameters = self.generator.parameters)

    @symbolic
    def generate(self, n_samples = None, noise = None):
        assert (noise is None) != (n_samples is None), "You must specify either noise or n_samples."
        if noise is None:
            noise = self.rng.normal(size=(n_samples, self.noise_dim))
        return self.generator(noise)