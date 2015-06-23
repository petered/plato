from general.test_mode import is_test_mode
from plato.tools.lstm import AutoencodingLSTM
from plato.tools.optimizers import AdaMax
from utils.bureaucracy import minibatch_iterate
from utils.datasets.books import read_the_bible, read_book
import numpy as np
from utils.tools.processors import OneHotEncoding


def demo_lstm_novelist(
        book = 'bible',
        n_hidden = 400,
        verse_duration = 40,
        generation_duration = 200,
        generate_every = 200,
        max_len = None,
        n_epochs = 1,
        seed = None,
        ):
    """
    An LSTM-Autoencoder learns the Bible, and can spontaniously produce biblical-ish verses.

    :param n_hidden: Number of hidden/memory units in LSTM
    :param verse_duration: Number of Backprop-Through-Time steps to do.
    :param generation_duration: Number of characters to generate with each sample.
    :param generate_every: Generate every N training iterations
    :param max_len: Truncate the text to this length.
    :param n_epochs: Number of passes through the bible to make.
    :param seed: Random Seed (None to use God's chosen seed).
    :return:
    """

    if is_test_mode():
        n_hidden=10
        verse_duration=3
        generation_duration=5
        max_len = 40

    rng = np.random.RandomState(seed)
    text = read_book(book, max_characters=max_len)

    onehot_text, decode_key = text_to_onehot(text)
    n_char = onehot_text.shape[1]

    the_prophet = AutoencodingLSTM(n_input=n_char, n_hidden=n_hidden,
        initializer_fcn=lambda shape: 0.01*rng.randn(*shape))

    training_fcn = the_prophet.get_training_function(optimizer=AdaMax(alpha = 0.01), update_states=True).compile()
    generating_fcn = the_prophet.get_generation_function(stochastic=True).compile()

    def prime_and_generate(n_steps, primer = ''):
        onehot_primer, _ = text_to_onehot(primer, decode_key)
        onehot_gen, = generating_fcn(onehot_primer, n_steps)
        gen = onehot_to_text(onehot_gen, decode_key)
        return '%s%s' % (primer, gen)

    print prime_and_generate(primer = 'In the beginning, ', n_steps = 100)

    for i, verse in enumerate(minibatch_iterate(onehot_text, minibatch_size=verse_duration, n_epochs=n_epochs)):
        if i % generate_every == 0:
            print 'Iteration %s:\n  ' % i + prime_and_generate(n_steps = 100)
        training_fcn(verse)

    trained_verses, _, _ = generating_fcn(generation_duration)
    display_generated('Final', onehot_to_text(trained_verses, decode_key))


def display_generated(title, text):
    print '%s %s %s\n%s\n%s' % ('='*10, title, '='*10, text, '='*30)


def text_to_onehot(text, decode_key = None):
    """
    :param text: A string of length N
    :return: (onehot, decode_key)
        onehot: A shape (N, n_unique_characters) array representing the one-hot encoding of each character.
        decode_key: The key translating columns of the onehot matrix back to characters.
    """
    text_array = np.array(text, 'c')
    if decode_key is None:
        decode_key, assignments = np.unique(text_array, return_inverse=True)
    else:
        assignments = np.searchsorted(decode_key, text_array)
    onehot = OneHotEncoding(n_classes=len(decode_key), dtype = np.float32)(assignments)
    return onehot, decode_key


def onehot_to_text(onehot, decode_key):
    assignments = np.argmax(onehot, axis = 1)
    text = decode_key[assignments].tostring()
    return text


EXPERIMENTS = dict()

EXPERIMENTS['learn_bible'] = lambda: demo_lstm_novelist(book = 'bible')

EXPERIMENTS['learn_fifty_shades'] = lambda: demo_lstm_novelist(book = 'fifty_shades_of_grey')


if __name__ == '__main__':

    EXPERIMENTS['learn_fifty_shades']
