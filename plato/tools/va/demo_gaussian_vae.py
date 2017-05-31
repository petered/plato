import numpy as np
from artemis.experiments.experiment_record import experiment_function
from artemis.experiments.ui import browse_experiments
from artemis.general.test_mode import is_test_mode
from artemis.ml.datasets.mnist import get_mnist_dataset
from artemis.ml.tools.iteration import minibatch_iterate
from artemis.plotting.db_plotting import dbplot
from plato.tools.optimization.optimizers import AdaMax
from plato.tools.va.gaussian_variational_autoencoder import GaussianVariationalAutoencoder


__author__ = 'peter'


@experiment_function
def demo_simple_vae_on_mnist(
        minibatch_size = 100,
        n_epochs = 2000,
        plot_interval = 100,
        calculation_interval = 500,
        z_dim = 2,
        hidden_sizes = [400, 200],
        learning_rate = 0.003,
        hidden_activation = 'softplus',
        binary_x = True,
        w_init_mag = 0.01,
        gaussian_min_var = None,
        manifold_grid_size = 11,
        manifold_grid_span = 2,
        seed = None
        ):
    """
    Train a Variational Autoencoder on MNIST and look at the samples it generates.
    """

    dataset = get_mnist_dataset(flat = True)
    training_data = dataset.training_set.input
    test_data = dataset.test_set.input

    if is_test_mode():
        n_epochs=1
        minibatch_size = 10
        training_data = training_data[:100]
        test_data = test_data[:100]

    model = GaussianVariationalAutoencoder(
        x_dim=training_data.shape[1],
        z_dim = z_dim,
        encoder_hidden_sizes = hidden_sizes,
        decoder_hidden_sizes = hidden_sizes[::-1],
        w_init_mag = w_init_mag,
        binary_data=binary_x,
        hidden_activation = hidden_activation,
        optimizer=AdaMax(alpha = learning_rate),
        gaussian_min_var = gaussian_min_var,
        rng = seed
        )

    training_fcn = model.train.compile()

    # For display, make functions to sample and represent the manifold.
    sampling_fcn = model.sample.compile()
    z_manifold_grid = np.array([x.flatten() for x in np.meshgrid(np.linspace(-manifold_grid_span, manifold_grid_span, manifold_grid_size),
        np.linspace(-manifold_grid_span, manifold_grid_span, manifold_grid_size))]+[np.zeros(manifold_grid_size**2)]*(z_dim-2)).T
    decoder_mean_fcn = model.decode.compile(fixed_args = dict(z = z_manifold_grid))
    lower_bound_fcn = model.compute_lower_bound.compile()

    for i, minibatch in enumerate(minibatch_iterate(training_data, minibatch_size=minibatch_size, n_epochs=n_epochs)):

        training_fcn(minibatch)

        if i % plot_interval == 0:
            samples = sampling_fcn(25).reshape(5, 5, 28, 28)
            dbplot(samples, 'Samples from Model')
            if binary_x:
                manifold_means = decoder_mean_fcn()
            else:
                manifold_means, _ = decoder_mean_fcn()
            dbplot(manifold_means.reshape(manifold_grid_size, manifold_grid_size, 28, 28), 'First 2-dimensions of manifold.')
        if i % calculation_interval == 0:
            training_lower_bound = lower_bound_fcn(training_data)
            test_lower_bound = lower_bound_fcn(test_data)
            print 'Epoch: %s, Training Lower Bound: %s, Test Lower bound: %s' % \
                (i*minibatch_size/float(len(training_data)), training_lower_bound, test_lower_bound)


# Try encoding MNIST with a variational autoencoder.
demo_simple_vae_on_mnist.add_variant('mnist-vae-20d-binary_in', z_dim = 20, hidden_sizes = [200], binary_x = True)
# Looks good.  Within about 20 epochs we're getting reasonablish samples, lower bound of -107.


# Try encoding MNIST with a variational autoencoder, this time treating the input as a continuous variable
demo_simple_vae_on_mnist.add_variant('mnist-vae-20d-continuous_in', z_dim = 20, hidden_sizes = [200], binary_x = False, gaussian_min_var = 0.01)
# Need to set minimum variance.  Recognieseable digits come out, but then instabilities.


demo_simple_vae_on_mnist.add_variant('mnist-vae-2latent', z_dim = 2, hidden_sizes = [400, 200], binary_x = True)


if __name__ == '__main__':
    browse_experiments()
