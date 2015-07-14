from plato.interfaces.decorators import symbolic_standard, symbolic_updater
import numpy as np
__author__ = 'peter'



class PatrickAutoencoder(object):

    def __init__(self, optimizer):


    @property
    def parameters(self):
        """
        :return: A list of theano shared variables.
        """

    @symbolic_updater
    def train(self, y_samples, a_samples):
        """
        :param y_samples:
        :param a_samples:
        :return: A list of updates
        """

        lower_bound = ...
        updates = self.optimizer(cost = -lower_bound, parameters = self.parameters)
        return updates

    @symbolic_standard
    def infer_x(self, y_samples, a_samples):
        """
        Take in samples of y, and masks
        :param y_samples:
        :param a_samples:
        :return: A reconstructed x
        """

        return (mu_x, sigma_x), []  # No updates


def train_my_model(
        minibatch_size = 100,
        n_epochs = 10,

        ):

    p = PatrickAutoencoder()

    observations, masks, ground_truth = get_data()
    # observations has shape (n_samples, n_dims)
    # masks is a boolean array with the same shape

    train_fcn = p.train.compile()
    inference_fcn = p.infer_x.compile()

    n_samples = len(observations)

    n_iterations = n_epochs*n_samples/minibatch_size
    for i in xrange(n_iterations):
        if i % 100 == 0:
            """ Somehow evaluate performance with inference fcn """
            mu, sigm = inference_fcn()

        ixs = np.arange(i, i+minibatch_size) % n_samples
        train_fcn(observations[ixs], masks[ixs])


if __name__ == '__main__':

    train_my_model()
