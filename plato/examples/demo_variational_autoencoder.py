from plato.tools.optimizers import AdaMax
from plato.tools.variational_autoencoder import VariationalAutoencoder, \
    EncoderDecoderNetworks
from plotting.db_plotting import dbplot
from utils.bureaucracy import minibatch_iterate
from utils.datasets.mnist import get_mnist_dataset
import numpy as np

__author__ = 'peter'


def demo_variational_autoencoder(
        minibatch_size = 100,
        n_epochs = 2000,
        plot_interval = 100,
        ):

    data = get_mnist_dataset(flat = True).training_set.input

    model = VariationalAutoencoder(
        pq_pair = EncoderDecoderNetworks(
            x_dim=data.shape[1],
            z_dim = 20,
            encoder_hidden_sizes = [200],
            decoder_hidden_sizes = [200],
            w_init = lambda n_in, n_out: 0.01*np.random.randn(n_in, n_out),
            x_distribution='bernoulli',
            z_distribution='gaussian',
            hidden_activation = 'rect-lin'
            ),
        optimizer=AdaMax(alpha = 0.01)
        )

    training_fcn = model.train.compile(mode = 'tr')

    sampling_fcn = model.sample.compile()

    for i, minibatch in enumerate(minibatch_iterate(data, minibatch_size=minibatch_size, n_epochs=n_epochs)):

        training_fcn(minibatch)

        if i % plot_interval == 0:
            samples = sampling_fcn(25).reshape(5, 5, 28, 28)
            dbplot(samples, 'Samples from Model')
            dbplot(model.pq_pair.p_net.parameters[-2].get_value()[:25].reshape(-1, 28, 28), 'dec')
            dbplot(model.pq_pair.q_net.parameters[0].get_value().T[:25].reshape(-1, 28, 28), 'enc')


if __name__ == '__main__':

    demo_variational_autoencoder()
