from plato.tools.dbn import DeepBeliefNet
from plato.tools.networks import StochasticLayer, FullyConnectedBridge
import numpy as np
from utils.datasets.mnist import get_mnist_dataset

__author__ = 'peter'


def demo_dbn_mnist():
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

    dataset = get_mnist_dataset().process_with(inputs_processor=lambda (x, ): (x.reshape(x.shape[0], -1), ))

    w_init = lambda n_in, n_out: 0.01 * np.random.randn(n_in, n_out)

    dbn = DeepBeliefNet(
        layers = {
            'vis': StochasticLayer('bernoulli'),
            'hid': StochasticLayer('bernoulli'),
            'ass': StochasticLayer('bernoulli'),
            'lab': StochasticLayer('bernoulli'),
            },
        bridges = {
            ('vis', 'hid'): FullyConnectedBridge(w = w_init(784, 500), b_rev = 0),
            ('hid', 'ass'): FullyConnectedBridge(w = w_init(500, 500), b_rev = 0),
            ('lab', 'ass'): FullyConnectedBridge(w = w_init(10, 500), b_rev = 0)
            }
        )

    # Compile the functions you're gonna use.
    train_first_layer = dbn.get_constrastive_divergence_function(visible_layers = 'vis', hidden_layers='hid', n_gibbs = 1, persistent=True).compile()
    # train_second_layer = dbn.get_constrastive_divergence_function(visible_layers=('vis', 'lab'), hidden_layers='ass', n_gibbs=1, persistent=True).compile()
    # predict_label = dbn.get_inference_function(input_layers = 'vis', output_layers='lab').compile()

    for _, visible_data, _ in dataset.training_set.minibatch_iterator(minibatch_size = minibatch_size, epochs = 10, single_channel = True):
        train_first_layer(visible_data)
        print 'Ran'
        # stream.update()


if __name__ == '__main__':

    demo_dbn_mnist()
