from plato.tools.dbn import simple_rbm
from plato.tools.networks import StochasticLayer, FullyConnectedBridge
from utils.datasets.mnist import get_mnist_dataset

__author__ = 'peter'



def demo_rbm():
    minibatch_size = 10

    data, _, _, _ = get_mnist_dataset().xyxy

    rbm = simple_rbm(
        visible_layer = StochasticLayer('bernoulli'),
        bridge=FullyConnectedBridge(w = 0.01*np.random.randn(784, 500)),
        hidden_layer = StochasticLayer('bernoulli')
        )

    train_function = rbm.get_training_fcn(n_gibbs = 1, persistent = True)
    sampling_function