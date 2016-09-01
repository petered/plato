from artemis.plotting.db_plotting import dbplot
from plato.tools.gan.gan import GenerativeAdversarialNetwork
from plato.tools.mlp.mlp import MultiLayerPerceptron
from plato.tools.optimization.optimizers import AdaMax
from utils.datasets.mnist import get_mnist_dataset
from utils.tools.iteration import minibatch_iterate

__author__ = 'peter'


def demo_gan_mnist(n_epochs = 20, minibatch_size = 20, n_discriminator_steps=1, noise_dim = 10, plot_period = 100, rng = 1234):
    """
    Train a Generative Adversarial network on MNIST data, showing generated samples as training progresses.

    :param n_epochs: Number of epochs to train
    :param minibatch_size: Size of minibatch to feed in each training iteration
    :param n_discriminator_steps: Number of steps training discriminator for every step of training generator
    :param noise_dim: Dimensionality of latent space (from which random samples are pulled)
    :param plot_period: Plot every N training iterations
    :param rng: Random number generator or seed
    """

    net = GenerativeAdversarialNetwork(
        discriminator = MultiLayerPerceptron.from_init(w_init=0.01, layer_sizes=[784, 100, 1], hidden_activation='relu', output_activation = 'sig', rng=rng),
        generator = MultiLayerPerceptron.from_init(w_init=0.1, layer_sizes=[noise_dim, 200, 784], hidden_activation='relu', output_activation = 'sig', rng=rng),
        noise_dim=noise_dim,
        optimizer=AdaMax(0.001),
        rng=rng
        )

    data = get_mnist_dataset(flat=True).training_set.input

    f_train_discriminator = net.train_discriminator.compile()
    f_train_generator = net.train_generator.compile()
    f_generate = net.generate.compile()

    for i, minibatch in enumerate(minibatch_iterate(data, n_epochs=n_epochs, minibatch_size=minibatch_size)):
        f_train_discriminator(minibatch)
        print 'Trained Discriminator'
        if i % n_discriminator_steps == n_discriminator_steps-1:
            f_train_generator(n_samples = minibatch_size)
            print 'Trained Generator'
        if i % plot_period == 0:
            samples = f_generate(n_samples=minibatch_size)
            dbplot(minibatch.reshape(-1, 28, 28), "Real")
            dbplot(samples.reshape(-1, 28, 28), "Counterfeit")
            print 'Disp'


if __name__ == '__main__':
    demo_gan_mnist()
