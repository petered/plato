from general.newline_writer import TextWrappingPrinter
from general.test_mode import is_test_mode
from plato.tools.lstm.long_short_term_memory import AutoencodingLSTM
from plato.tools.optimization.optimizers import AdaMax
from utils.tools.iteration import minibatch_iterate
from utils.datasets.books import read_book
import numpy as np
from utils.tools.processors import OneHotEncoding
import theano


def demo_lstm_novelist(
        book = 'bible',
        n_hidden = 400,
        verse_duration = 20,
        generation_duration = 200,
        generate_every = 200,
        max_len = None,
        hidden_layer_type = 'tanh',
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
    :param seed: Random Seed
    :return:
    """

    if is_test_mode():
        n_hidden=10
        verse_duration=7
        generation_duration=5
        max_len = 40

    rng = np.random.RandomState(seed)
    text = read_book(book, max_characters=max_len)

    onehot_text, decode_key = text_to_onehot(text)
    n_char = onehot_text.shape[1]

    the_prophet = AutoencodingLSTM(n_input=n_char, n_hidden=n_hidden,
        initializer_fcn=lambda shape: 0.01*rng.randn(*shape), hidden_layer_type = hidden_layer_type)

    training_fcn = the_prophet.get_training_function(optimizer=AdaMax(alpha = 0.01), update_states=True).compile()
    generating_fcn = the_prophet.get_generation_function(stochastic=True).compile()

    printer = TextWrappingPrinter(newline_every=100)

    def prime_and_generate(n_steps, primer = ''):
        onehot_primer, _ = text_to_onehot(primer, decode_key)
        onehot_gen, = generating_fcn(onehot_primer, n_steps)
        gen = onehot_to_text(onehot_gen, decode_key)
        return '%s%s' % (primer, gen)

    prime_and_generate(generation_duration, 'In the beginning, ')

    for i, verse in enumerate(minibatch_iterate(onehot_text, minibatch_size=verse_duration, n_epochs=n_epochs)):
        if i % generate_every == 0:
            printer.write('[iter %s]%s' % (i, prime_and_generate(n_steps = generation_duration), ))
        training_fcn(verse)

    printer.write('[iter %s]%s' % (i, prime_and_generate(n_steps = generation_duration), ))


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
    onehot = OneHotEncoding(n_classes=len(decode_key), dtype = theano.config.floatX)(assignments)
    return onehot, decode_key


def onehot_to_text(onehot, decode_key):
    assignments = np.argmax(onehot, axis = 1)
    text = decode_key[assignments].tostring()
    return text


EXPERIMENTS = dict()

EXPERIMENTS['learn_bible'] = lambda: demo_lstm_novelist(book = 'bible')

EXPERIMENTS['learn_fifty_shades'] = lambda: demo_lstm_novelist(book = 'fifty_shades_of_grey', n_epochs = 4)


if __name__ == '__main__':

    EXPERIMENTS['learn_fifty_shades']()
