from fileman.experiment_record import run_experiment
from general.test_mode import set_test_mode
from plato.tools.rbm.demo_rbm_tutorial import demo_rbm_tutorial
from plato.tools.lstm.demo_long_short_term_memory import demo_lstm_novelist
from plato.tools.regressors.demo_mnist_regression import demo_mnist_online_regression
from plato.tools.va.demo_variational_autoencoder import demo_variational_autoencoder
from plato.tools.optimization.demo_compare_optimizers import get_experiments
from plato.examples.demo_prediction_example import compare_example_predictors
from plato.tools.mlp.demo_mnist_mlp import demo_mnist_mlp
from plato.tools.dbn.demo_dbn import demo_dbn_mnist
from plato.tools.rbm.demo_rbm import demo_rbm_mnist
from plato.tools.va.demo_gaussian_vae import demo_simple_vae_on_mnist
import plato.tools.dtp.demo_difference_target_propagation  # Demo fetches experiments so don't remove.
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


def test_demo_variational_autoencoder():
    demo_variational_autoencoder()


def test_demo_lstm():
    demo_lstm_novelist()


def test_demo_difference_target_prop():

    run_experiment('backprop-vs-dtp')
    run_experiment('standard-dtp')
    run_experiment('all-relu-LinDTP')


def test_demo_simple_vae_on_mnist():
    demo_simple_vae_on_mnist(binary_x=True)
    demo_simple_vae_on_mnist(binary_x=False)


def test_rbm_tutorial():
    demo_rbm_tutorial()


def test_demo_mnist_regression():
    demo_mnist_online_regression()


if __name__ == '__main__':
    set_test_mode(True)
    # test_demo_mnist_regression()
    # test_demo_difference_target_prop()
    # test_rbm_tutorial()
    # test_demo_simple_vae_on_mnist()
    # test_demo_lstm()
    # test_demo_variational_autoencoder()
    # test_demo_compare_optimizers()
    # test_demo_prediction_example()
    # test_demo_mnist_mlp()
    # test_demo_rbm_mnist()
    test_demo_dbn_mnist()
