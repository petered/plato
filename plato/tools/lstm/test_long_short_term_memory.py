import theano
import numpy as np

from artemis.general.test_mode import set_test_mode
from plato.tools.lstm.demo_long_short_term_memory import demo_lstm_novelist
from plato.tools.lstm.long_short_term_memory import AutoencodingLSTM
from plato.tools.optimization.optimizers import AdaMax
from artemis.ml.tools.iteration import minibatch_iterate
from artemis.ml.tools.processors import OneHotEncoding


__author__ = 'peter'


def test_autoencoding_lstm(
        width = 8,
        seed = 1234):

    data = get_bounce_data(width=width)
    encoder = OneHotEncoding(n_classes=width, dtype = theano.config.floatX)
    onehot_data = encoder(data)
    rng = np.random.RandomState(seed)
    aelstm = AutoencodingLSTM(n_input = 8, n_hidden=50, initializer_fcn = lambda shape: 0.01*rng.randn(*shape))

    gen_fcn = aelstm.get_generation_function(maintain_state=True, rng = rng).compile(add_test_values = True)
    train_fcn = aelstm.get_training_function(update_states=True, optimizer = AdaMax(alpha = 0.1)).compile(add_test_values = True)

    def prime_and_gen(primer, n_steps):
        onehot_primer = encoder(np.array(primer))
        onehot_generated, = gen_fcn(onehot_primer, n_steps)
        generated = encoder.inverse(onehot_generated)
        return generated

    initial_seq = prime_and_gen([0, 1, 2, 3, 4], 11)
    print initial_seq

    # Test empty, one-length primers
    prime_and_gen([], 2)
    prime_and_gen([0], 2)

    print 'Training....'
    for d in minibatch_iterate(onehot_data, minibatch_size=3, n_epochs=400):
        train_fcn(d)
    print 'Done.'

    final_seq = prime_and_gen([0, 1, 2, 3, 4], 11)
    assert np.array_equal(final_seq, [5, 6, 7, 6, 5, 4, 3, 2, 1, 0, 1]), 'Bzzzz! It was %s' % (final_seq, )

    # Assert state is maintained
    seq = prime_and_gen([], 3)
    assert np.array_equal(seq, [2, 3, 4]), 'Bzzzz! It was %s' % (seq, )
    seq = prime_and_gen([5], 3)
    assert np.array_equal(seq, [6, 7, 6]), 'Bzzzz! It was %s' % (seq, )

    # Assert training does not interrupt generation state.
    train_fcn(d)
    seq = prime_and_gen([], 3)
    assert np.array_equal(seq, [5, 4, 3]), 'Bzzzz! It was %s' % (seq, )


def test_demo_lstm():
    demo_lstm_novelist()

if __name__ == '__main__':

    set_test_mode(True)
    test_autoencoding_lstm()
    test_demo_lstm()


def get_bounce_data(width = 8, n_rounds = 1, onehot = False):
    """
    Data bounes between a max and min value.

    [0,1,2,3,2,1,0,1,2,3,2,1,0,...]

    :param period:
    :param n_rounds:
    :param onehot:
    :return:
    """

    period = width*2 - 2
    n_samples = period * n_rounds

    x = np.arange(n_samples)

    x %= period
    x[x>=width] = period - x[x>=width]

    if onehot:
        onehot_x = np.zeros((n_samples, width))
        onehot_x[np.arange(n_samples), x] = 1
        return onehot_x
    else:
        return x