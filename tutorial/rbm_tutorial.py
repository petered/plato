from general.test_mode import is_test_mode
from plotting.db_plotting import dbplot
from utils.bureaucracy import minibatch_iterate
from utils.datasets.mnist import get_mnist_dataset
import numpy as np

__author__ = 'peter'


def demo_rbm_tutorial(
        eta = 0.01,
        n_hidden = 500,
        n_samples = None,
        minibatch_size = 10,
        plot_interval = 10,
        w_init_mag = 0.01,
        n_epochs = 1,
        persistent = False,
        seed = None
        ):
    """
    This tutorial trains a standard binary-binary RBM on MNIST, and allows you to view the weights and negative sampling
    chain.

    Note:
    For simplicity, it uses hidden/visible samples to compute the gradient.  It's actually better to use the hidden
    probabilities.
    """
    if is_test_mode():
        n_samples=50
        n_epochs=1
        plot_interval=50
        n_hidden = 10

    data = get_mnist_dataset(flat = True).training_set.input[:n_samples]
    n_visible = data.shape[1]
    rng = np.random.RandomState(seed)
    activation = lambda x: (1./(1+np.exp(-x)) > rng.rand(*x.shape)).astype(float)

    w = w_init_mag*np.random.randn(n_visible, n_hidden)
    b_hid = np.zeros(n_hidden)
    b_vis = np.zeros(n_visible)

    if persistent:
        hid_sleep_state = np.random.rand(minibatch_size, n_hidden)

    for i, vis_wake_state in enumerate(minibatch_iterate(data, n_epochs = n_epochs, minibatch_size=minibatch_size)):
        hid_wake_state = activation(vis_wake_state.dot(w)+b_hid)
        if not persistent:
            hid_sleep_state = hid_wake_state
        vis_sleep_state = activation(hid_sleep_state.dot(w.T)+b_vis)
        hid_sleep_state = activation(vis_sleep_state.dot(w)+b_hid)

        # Update Parameters
        w_grad = (vis_wake_state.T.dot(hid_wake_state) - vis_sleep_state.T.dot(hid_sleep_state))/float(minibatch_size)
        w += w_grad * eta
        b_vis_grad = np.mean(vis_wake_state, axis = 0) - np.mean(vis_sleep_state, axis = 0)
        b_vis += b_vis_grad * eta
        b_hid_grad = np.mean(hid_wake_state, axis = 0) - np.mean(hid_sleep_state, axis = 0)
        b_hid += b_hid_grad * eta

        if i % plot_interval == 0:
            dbplot(w.T[:100].reshape(-1, 28, 28), 'weights')
            dbplot(vis_sleep_state.reshape(-1, 28, 28), 'dreams')
            print 'Sample %s' % i


if __name__ == '__main__':
    demo_rbm_tutorial()
