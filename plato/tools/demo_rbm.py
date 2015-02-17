from plato.tools.rbm import simple_rbm
from plato.tools.networks import StochasticLayer, FullyConnectedBridge, ConvolutionalBridge
from plato.tools.optimizers import SimpleGradientDescent
from plotting.live_plotting import LiveStream
import theano
from utils.datasets.mnist import get_mnist_dataset
import numpy as np

__author__ = 'peter'


def demo_rbm(architecture = 'conv'):
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
    minibatch_size = 9

    dataset = get_mnist_dataset().process_with(inputs_processor=lambda (x, ): (x[:, None, :, :], ))

    rbm = simple_rbm(
        visible_layer = StochasticLayer('bernoulli'),
        # bridge=FullyConnectedBridge(w = 0.001*np.random.randn(28*28, 500), b = 0, b_rev = 0),
        bridge=ConvolutionalBridge(w = 0.01*np.random.randn(12, 1, 7, 7), b = 0, b_rev = 0),
        hidden_layer = StochasticLayer('bernoulli')
        )

    train_function = rbm.get_training_fcn(n_gibbs = 4, persistent = True, optimizer = SimpleGradientDescent(eta = 0.01)).compile(debug_getter = 'locals')
    sampling_function = rbm.get_free_sampling_fcn(init_visible_state = np.random.randn(* ((minibatch_size, )+(dataset.training_set.input.shape[1:]))), return_smooth_visible = True).compile()

    # Setup Plotting
    def get_plot_vars():
        lv = train_function.get_debug_values()
        return {
            'full': lambda: {
                'hidden-neg-chain': lv.sleep_hidden.reshape((-1, 25, 20)),
                'visible-neg-chain': lv.hidden_layer.smooth(lv.bridge.reverse(lv.sleep_hidden)).reshape((-1, 28, 28)),
                'w': lv.bridge.parameters[0].get_value().T[:25].reshape((-1, 28, 28)),
                'b': lv.bridge.parameters[1].get_value().reshape((25, 20)),
                'b_rev': lv.bridge.parameters[2].get_value().reshape((28, 28)),
                'visible-sample': visible_samples.reshape((-1, 28, 28))
                },
            'conv': lambda: {
                'hidden-neg-chain': lv.sleep_hidden,
                'visible-neg-chain': lv.hidden_layer.smooth(lv.bridge.reverse(lv.sleep_hidden)),
                'w': lv.bridge.parameters[0].get_value(),
                'b': lv.bridge.parameters[1].get_value(),
                'b_rev': lv.bridge.parameters[2].get_value(),
                'visible-sample': visible_samples
                }
            }[architecture]()

    stream = LiveStream(get_plot_vars, update_every=10)

    for _, visible_data, _ in dataset.training_set.minibatch_iterator(minibatch_size = minibatch_size, epochs = 10, single_channel = True):
        visible_samples, _ = sampling_function()
        train_function(visible_data)
        stream.update()


if __name__ == '__main__':

    demo_rbm()
