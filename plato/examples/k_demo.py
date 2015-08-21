from general.test_mode import is_test_mode
from plato.tools.optimization.optimizers import AdaMax, GradientDescent
from plato.tools.va.gaussian_variational_autoencoder import GaussianVariationalAutoencoder
from plotting.db_plotting import dbplot
from utils.bureaucracy import minibatch_iterate
import numpy as np

__author__ = 'peter'


def generate_tones(tones, sample_length, sampling_freq = 8192):

    t = np.linspace(0, sample_length, sampling_freq*sample_length)
    x = np.sin(t*np.array(tones)[:, None]*2*np.pi)
    return x


def demo_learn_tones(
        sample_length = 0.01,
        sampling_freq = 8192,
        tones = 440*(2**np.linspace(0, 1, 12)),
        n_epochs = 200000,
        plot_interval = 100,
        calculation_interval = 500,
        z_dim = 5,
        hidden_sizes = [500],
        learning_rate = 0.0003,
        hidden_activation = 'lin',
        w_init_mag = 0.01,
        minibatch_size='full',
        manifold_grid_size = 3,
        manifold_grid_span = 2,
        seed = None
        ):
    """
    Train a Variational Autoencoder on MNIST and look at the samples it generates.
    """

    training_data = test_data = generate_tones(tones = tones, sample_length=sample_length, sampling_freq=sampling_freq)

    if minibatch_size == 'full':
        minibatch_size = len(training_data)

    dbplot(training_data, 'Training Data')

    # dataset = get_mnist_dataset(flat = True)
    # training_data = dataset.training_set.input
    # test_data = dataset.test_set.input

    if is_test_mode():
        n_epochs=1
        training_data = training_data[:100]
        test_data = test_data[:100]

    model = GaussianVariationalAutoencoder(
        x_dim=training_data.shape[1],
        z_dim = z_dim,
        encoder_hidden_sizes = hidden_sizes,
        decoder_hidden_sizes = hidden_sizes[::-1],
        w_init_mag = w_init_mag,
        binary_data=False,
        hidden_activation = hidden_activation,
        optimizer = AdaMax(alpha = learning_rate),
        gaussian_min_var=0.01,
        rng = seed
        )

    training_fcn = model.train.compile()

    # For display, make functions to sample and represent the manifold.
    sampling_fcn = model.sample.compile()
    z_manifold_grid = np.array([x.flatten() for x in np.meshgrid(np.linspace(-manifold_grid_span, manifold_grid_span, manifold_grid_size),
        np.linspace(-manifold_grid_span, manifold_grid_span, manifold_grid_size))]+[np.zeros(manifold_grid_size**2)]*(z_dim-2)).T
    decoder_mean_fcn = model.decode.compile(fixed_args = dict(z = z_manifold_grid))
    lower_bound_fcn = model.compute_lower_bound.compile()

    recon_fcn = model.recon.compile()

    for i, minibatch in enumerate(minibatch_iterate(training_data, minibatch_size=minibatch_size, n_epochs=n_epochs)):

        training_fcn(minibatch)

        if i % plot_interval == 0:
            samples = sampling_fcn(12)
            dbplot(samples, 'Samples from Model')
            dbplot(recon_fcn(samples), 'Stochastic Reconstructions')
            manifold_means, _ = decoder_mean_fcn()
            dbplot(manifold_means.reshape(manifold_grid_size, manifold_grid_size, 1, training_data.shape[1]), 'First 2-dimensions of manifold.')
        if i % calculation_interval == 0:
            training_lower_bound = lower_bound_fcn(training_data)
            test_lower_bound = lower_bound_fcn(test_data)
            print 'Epoch: %s, Training Lower Bound: %s, Test Lower bound: %s' % \
                (i*minibatch_size/float(len(training_data)), training_lower_bound, test_lower_bound)


if __name__ == '__main__':

    demo_learn_tones()
