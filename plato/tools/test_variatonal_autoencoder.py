from plato.tools.variational_autoencoder import VariationalAutoencoder, EncoderDecoderNetworks
from utils.bureaucracy import minibatch_iterate
from utils.datasets.synthetic_clusters import get_synthetic_clusters_dataset

__author__ = 'peter'


def test_variational_autoencoder():

    dataset = get_synthetic_clusters_dataset()

    model = VariationalAutoencoder(
        pq_pair=EncoderDecoderNetworks(
            x_dim = dataset.input_shape[0],
            z_dim = 2,
            encoder_hidden_sizes=[50],
            decoder_hidden_sizes=[50]
            )
        )

    train_fcn = model.train.compile()
    log_prob_fcn = model.log_prob_data.compile()

    initial_log_prob = log_prob_fcn(dataset.test_set.input, 20)

    for minibatch in minibatch_iterate(dataset.training_set.input, minibatch_size = 100, n_epochs=2)

        train_fcn(minibatch)

    final_log_prob = log_prob_fcn(dataset.test_set.input, 20)

    assert final_log_prob > initial_log_prob


if __name__ == '__main__':

