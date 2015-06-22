from plato.tools.lstm import AutoencodingLSTM
from utils.bureaucracy import minibatch_iterate
from utils.datasets.bounce_data import get_bounce_data
import numpy as np

__author__ = 'peter'


def test_autoencoding_lstm(seed = 1234):

    data = get_bounce_data(period=14, onehot = True).astype('float32')

    rng = np.random.RandomState(seed)

    aelstm = AutoencodingLSTM(n_input = 8, n_hidden=20, initializer_fcn = lambda shape: 0.01*rng.randn(*shape))

    gen_fcn = aelstm.get_generation_function().compile()
    train_fcn = aelstm.get_training_function().compile()

    initial_seq, = gen_fcn(14)
    print np.argmax(initial_seq, axis = 1)

    for d in minibatch_iterate(data, minibatch_size=5, n_epochs=100):

        train_fcn(d)

    final_seq, = gen_fcn(14)

    print np.argmax(final_seq, axis = 1)


if __name__ == '__main__':

    test_autoencoding_lstm()
