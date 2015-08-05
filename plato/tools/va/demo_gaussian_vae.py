from general.test_mode import is_test_mode
from plato.tools.optimization.optimizers import AdaMax
from plato.tools.va.gaussian_variational_autoencoder import GaussianVariationalAutoencoder
from plotting.db_plotting import dbplot
from utils.bureaucracy import minibatch_iterate
from utils.datasets.mnist import get_mnist_dataset
import numpy as np

__author__ = 'peter'


def demo_simple_vae_on_mnist(
        minibatch_size = 100,
        n_epochs = 2000,
        plot_interval = 100,
        z_dim = 20,
        hidden_sizes = [200],
        learning_rate = 0.003,
        hidden_activation = 'relu',
        binary_x = True,
        w_init_mag = 0.01,
        manifold_grid_size = 11,
        manifold_grid_span = 2,
        seed = None
        ):
    """
    Train a Variational Autoencoder on MNIST and look at the samples it generates.
    :param minibatch_size: Number of elements in the minibatch
    :param n_epochs: Number of passes through dataset
    :param plot_interval: Plot every x iterations
    """

    data = get_mnist_dataset(flat = True).training_set.input

    if is_test_mode():
        n_epochs=1
        minibatch_size = 10
        data = data[:100]

    model = GaussianVariationalAutoencoder(
        x_dim=data.shape[1],
        z_dim = z_dim,
        encoder_hidden_sizes = hidden_sizes,
        decoder_hidden_sizes = hidden_sizes[::-1],
        w_init_mag = w_init_mag,
        binary_data=binary_x,
        hidden_activation = hidden_activation,
        optimizer=AdaMax(alpha = learning_rate),
        rng = seed
        )

    training_fcn = model.train.compile()

    # For display, make functions to sample and represent the manifold.
    sampling_fcn = model.sample.compile()
    z_manifold_grid = np.array([x.flatten() for x in np.meshgrid(np.linspace(-manifold_grid_span, manifold_grid_span, manifold_grid_size),
        np.linspace(-manifold_grid_span, manifold_grid_span, manifold_grid_size))]+[np.zeros(manifold_grid_size**2)]*(z_dim-2)).T
    decoder_mean_fcn = model.decode.compile(fixed_args = dict(z = z_manifold_grid))

    for i, minibatch in enumerate(minibatch_iterate(data, minibatch_size=minibatch_size, n_epochs=n_epochs)):

        training_fcn(minibatch)

        if i % plot_interval == 0:
            print 'Epoch %s' % (i*minibatch_size/50000., )
            samples = sampling_fcn(25).reshape(5, 5, 28, 28)
            dbplot(samples, 'Samples from Model')
            if binary_x:
                manifold_means = decoder_mean_fcn()
            else:
                manifold_means, _ = decoder_mean_fcn()
            dbplot(manifold_means.reshape(manifold_grid_size, manifold_grid_size, 28, 28), 'First 2-dimensions of manifold.')


if __name__ == '__main__':

    demo_simple_vae_on_mnist()
