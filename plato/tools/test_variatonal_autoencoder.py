from plato.tools.optimizers import AdaMax
from plato.tools.variational_autoencoder import VariationalAutoencoder, EncoderDecoderNetworks
from utils.bureaucracy import minibatch_iterate
from utils.datasets.synthetic_clusters import get_synthetic_clusters_dataset
import numpy as np
import pytest

__author__ = 'peter'


def mean_closest_match(x, y, distance_measure = 'euclidian'):
    """

    :param x: An (n_samples_x, n_dims) array
    :param y: An (n_samples_y, n_dims) array
    :return: A scalar indicating the mean closest match of an element in x, to an element in y
    """
    dist_fcn = {
        'L1': lambda x, y: np.sum(np.abs(x[:, None, :]-y[None, :, :]), axis = 2),
        'euclidian': lambda x, y: np.sqrt(np.sum((x[:, None, :]-y[None, :, :])**2, axis = 2))
        }[distance_measure]
    distances = dist_fcn(x, y)  # (n_samples_x, n_samples_y)
    return np.mean(np.min(distances, axis = 1))


@pytest.mark.skipif(True, reason = 'Fails in pytest due to some weird reference-counter bug in theano.')
def test_variational_autoencoder():
    """
    Just test that after training, samples are closer to the test data than they are before training.
    """

    dataset = get_synthetic_clusters_dataset()
    rng = np.random.RandomState(1234)
    model = VariationalAutoencoder(
        pq_pair=EncoderDecoderNetworks(
            x_dim = dataset.input_shape[0],
            z_dim = 2,
            encoder_hidden_sizes=[],
            decoder_hidden_sizes=[],
            w_init=lambda n_in, n_out: 0.01*rng.randn(n_in, n_out),
            ),
        optimizer=AdaMax(alpha = 0.1),
        rng = rng
        )
    train_fcn = model.train.compile()
    gen_fcn = model.sample.compile()
    initial_mcm = mean_closest_match(gen_fcn(100), dataset.test_set.input, 'L1')
    for minibatch in minibatch_iterate(dataset.training_set.input, minibatch_size = 10, n_epochs=1):
        train_fcn(minibatch)
    final_mcm = mean_closest_match(gen_fcn(100), dataset.test_set.input, 'L1')
    assert final_mcm < initial_mcm / 2


if __name__ == '__main__':

    test_variational_autoencoder()
