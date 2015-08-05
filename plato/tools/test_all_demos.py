from general.test_mode import set_test_mode
from plato.tools.lstm.demo_long_short_term_memory import demo_lstm_novelist
from plato.tools.va import demo_variational_autoencoder
from plato.tools.optimization.demo_compare_optimizers import get_experiments
from plato.examples.demo_prediction_example import compare_example_predictors
from plato.tools.mlp.demo_mnist_mlp import demo_mnist_mlp
from plato.tools.dbn.demo_dbn import demo_dbn_mnist
from plato.tools.rbm.demo_rbm import demo_rbm_mnist
from plato.tools.dtp.demo_difference_target_propagation import EXPERIMENTS as DTP_EXPERIMENTS
from plato.tools.va.demo_gaussian_vae import demo_simple_vae_on_mnist
import pytest
__author__ = 'peter'


def test_demo_compare_optimizers():

    for exp_name, exp in get_experiments().iteritems():
        print 'Running %s' % exp_name
        exp()


def test_demo_mnist_mlp():
    demo_mnist_mlp()


def test_demo_dbn_mnist():
    demo_dbn_mnist(plot = True)


def test_demo_rbm_mnist():
    demo_rbm_mnist(plot = True)


def test_demo_prediction_example():
    compare_example_predictors()


@pytest.mark.skipif(True, reason = 'Fails in pytest due to some weird reference-counter bug in theano.')
def test_demo_variational_autoencoder():
    demo_variational_autoencoder()


@pytest.mark.skipif(True, reason = 'Fails in pytest due to some weird reference-counter bug in theano.')
def test_demo_lstm():
    demo_lstm_novelist()


def test_demo_difference_target_prop():

    for exp, val in DTP_EXPERIMENTS.iteritems():
        print 'Running %s' % (exp, )
        val()


def test_demo_simple_vae_on_mnist():
    demo_simple_vae_on_mnist(binary_x=True)
    demo_simple_vae_on_mnist(binary_x=False)


if __name__ == '__main__':
    set_test_mode(True)
    test_demo_simple_vae_on_mnist()
    test_demo_difference_target_prop()
    test_demo_lstm()
    test_demo_variational_autoencoder()
    test_demo_compare_optimizers()
    test_demo_prediction_example()
    test_demo_mnist_mlp()
    test_demo_rbm_mnist()
    test_demo_dbn_mnist()
