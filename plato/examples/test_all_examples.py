from general.test_mode import set_test_mode
from plato.examples.demo_lstm import demo_lstm_novelist
from plato.examples.demo_variational_autoencoder import demo_variational_autoencoder
from plato.examples.demo_compare_optimizers import get_experiments
from plato.examples.demo_prediction_example import compare_example_predictors
from plato.examples.demo_mnist_mlp import demo_mnist_mlp
from plato.examples.demo_dbn import demo_dbn_mnist
from plato.examples.demo_rbm import demo_rbm_mnist
import pytest
__author__ = 'peter'


def test_demo_compare_optimizers():

    for exp_name, exp in get_experiments().iteritems():
        print 'Running %s' % exp_name
        exp()


def test_demo_mnist_mlp():
    demo_mnist_mlp(test_mode = True)


def test_demo_dbn_mnist():
    demo_dbn_mnist(plot = True, test_mode = True)


def test_demo_rbm_mnist():
    demo_rbm_mnist(plot = True)


def test_demo_prediction_example():
    compare_example_predictors(test_mode = True)


@pytest.mark.skipif(True, reason = 'Fails in pytest due to some weird reference-counter bug in theano.')
def test_demo_variational_autoencoder():
    demo_variational_autoencoder()


@pytest.mark.skipif(True, reason = 'Fails in pytest due to some weird reference-counter bug in theano.')
def test_demo_lstm():
    demo_lstm_novelist()


if __name__ == '__main__':
    set_test_mode(True)
    test_demo_lstm()
    test_demo_variational_autoencoder()
    test_demo_compare_optimizers()
    test_demo_prediction_example()
    test_demo_mnist_mlp()
    test_demo_rbm_mnist()
    test_demo_dbn_mnist()
