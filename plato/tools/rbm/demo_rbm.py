from general.should_be_builtins import bad_value
from general.test_mode import is_test_mode
from plato.core import EnableOmbniscence
from plato.tools.rbm.restricted_boltzmann_machine import simple_rbm
from plato.tools.rbm.rbm_parts import StochasticNonlinearity, FullyConnectedBridge
from plato.tools.optimization.optimizers import SimpleGradientDescent, AdaMax
from plotting.db_plotting import dbplot
import theano
from utils.tools.iteration import minibatch_iterate
from utils.datasets.mnist import get_mnist_dataset
import numpy as np

__author__ = 'peter'


def demo_rbm_mnist(
        vis_activation = 'bernoulli',
        hid_activation = 'bernoulli',
        n_hidden = 500,
        plot = True,
        eta = 0.01,
        optimizer = 'sgd',
        w_init_mag = 0.001,
        minibatch_size = 9,
        persistent = False,
        n_epochs = 100,
        plot_interval = 100,
        ):
    """
    In this demo we train an RBM on the MNIST input data (labels are ignored).  We plot the state of a markov chanin
    that is being simulaniously sampled from the RBM, and the parameters of the RBM.

    What you see:
    A plot will appear with 6 subplots.  The subplots are as follows:
    hidden-neg-chain: The activity of the hidden layer for each of the persistent CD chains for draewing negative samples.
    visible-neg-chain: The probabilities of the visible activations corresponding to the state of hidden-neg-chain.
    w: A subset of the weight vectors, reshaped to the shape of the input.
    b: The bias of the hidden units.
    b_rev: The bias of the visible units.
    visible-sample: The probabilities of the visible samples drawin from an independent free-sampling chain (outside the
        training function).

    As learning progresses, visible-neg-chain and visible-sample should increasingly resemble the data.
    """
    with EnableOmbniscence():

        if is_test_mode():
            n_epochs = 0.01

        data = get_mnist_dataset(flat = True).training_set.input

        rbm = simple_rbm(
            visible_layer = StochasticNonlinearity(vis_activation),
            bridge=FullyConnectedBridge(w = w_init_mag*np.random.randn(28*28, n_hidden).astype(theano.config.floatX), b=0, b_rev = 0),
            hidden_layer = StochasticNonlinearity(hid_activation)
            )

        optimizer = \
            SimpleGradientDescent(eta = eta) if optimizer == 'sgd' else \
            AdaMax(alpha=eta) if optimizer == 'adamax' else \
            bad_value(optimizer)

        train_function = rbm.get_training_fcn(n_gibbs = 1, persistent = persistent, optimizer = optimizer).compile()

        def plot_fcn():
            lv = train_function.locals()
            dbplot({
                'visible-pos-chain': lv['wake_visible'].reshape((-1, 28, 28)),
                'visible-neg-chain': lv['sleep_visible'].reshape((-1, 28, 28)),
                })

        for i, visible_data in enumerate(minibatch_iterate(data, minibatch_size=minibatch_size, n_epochs=n_epochs)):
            train_function(visible_data)
            if plot and i % plot_interval == 0:
                plot_fcn()


EXPERIMENTS = dict()

EXPERIMENTS['standard'] = lambda: demo_rbm_mnist(vis_activation='bernoulli', hid_activation='bernoulli', n_hidden=500, w_init_mag=0.01, eta = 0.01)

EXPERIMENTS['relu'] = lambda: demo_rbm_mnist(vis_activation='relu', hid_activation='relu', persistent = True, n_hidden=500, optimizer = 'adamax', w_init_mag=0.01, eta = 0.0001)


if __name__ == '__main__':

    experiment = 'relu'

    EXPERIMENTS[experiment]()
