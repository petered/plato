from plato.tools.dbn import simple_rbm
from plato.tools.networks import StochasticLayer, FullyConnectedBridge
from plato.tools.optimizers import SimpleGradientDescent
from plotting.showme import LiveStream
import theano
from utils.datasets.mnist import get_mnist_dataset
import numpy as np

__author__ = 'peter'


def demo_rbm():
    minibatch_size = 10

    dataset = get_mnist_dataset().process_with(inputs_processor=lambda (x, ): (x.reshape(x.shape[0], -1), ))

    rbm = simple_rbm(
        visible_layer = StochasticLayer('bernoulli'),
        bridge=FullyConnectedBridge(w = 0.01*np.random.randn(28*28, 500).astype(theano.config.floatX)),
        hidden_layer = StochasticLayer('bernoulli')
        )

    train_function = rbm.get_training_fcn(n_gibbs = 1, persistent = True, optimizer = SimpleGradientDescent(eta = 0.01)).compile()
    sampling_function = rbm.get_free_sampling_fcn(init_visible_state = np.random.randn(9, 28*28), return_smooth_visible = True).compile()

    stream = LiveStream(lambda: {
        # 'visible_sleep': train_function.locals['visible_sleep'].reshape(-1, 28, 28),
        # 'hidden_sleep': train_function.locals['hidden_sleep'].reshape(-1, 25, 20),
        # 'w': train_function.locals['bridge']._w.get_value().T[:25].reshape(-1, 28, 28)
        'visible': visible.reshape(-1, 28, 28),
        'hidden': hidden.reshape(-1, 25, 20),
        'w': rbm.vars['bridge']._w.get_value().T[:25].reshape(-1, 28, 28)
        }, update_every=10)
    for _, visible_data, _ in dataset.training_set.minibatch_iterator(minibatch_size = minibatch_size, epochs = 10, single_channel = True):
        visible, hidden = sampling_function()
        train_function(visible_data)
        stream.update()


if __name__ == '__main__':

    demo_rbm()
