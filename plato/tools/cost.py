from abc import ABCMeta, abstractmethod
from plato.interfaces.decorators import symbolic_stateless
import theano.tensor as tt

__author__ = 'peter'


class ICostFunction(object):

    __metaclass__ = ABCMeta

    @abstractmethod
    def __call__(self, actual, target):
        """
        :param actual: An (n_samples x ...) tensor representing the actual output
        :param target: An (n_samples x ...) tensor representing the target output
        :return: A symbolic scalar representing the cost
        """

@symbolic_stateless
def softmax_negative_log_likelihood(actual, target):
    """
    Do a softmax on the actual along axis 1 and then compute NLL
    """
    # Note: We do not just call negative_log_likelihood because the assert statement
    # in there can (maybe?) block the softmax-log optimization.
    normalized_actual = tt.nnet.softmax(actual)
    return -tt.log(actual[tt.arange(normalized_actual.shape[0]), target]).mean()


@symbolic_stateless
def negative_log_likelihood(actual, target):
    """
    :param actual: An (n_samples, n_labels) tensor where rows are normalized and actual[i,j] indicates the belief
        that on sample[i] the correct target is j.
    :param target: An (n_samples, ) tensor indicating the target label for each sample
    :return: The average (over samples) of the negative log-likelihood.
    """
    actual = tt.opt.assert_(actual, tt.all(abs(actual.sum(axis=1)-1) < 1e-7))  # Data must be normalized along axis 1.
    return -tt.log(actual[tt.arange(actual.shape[0]), target]).mean()


@symbolic_stateless
def mean_squared_error(actual, target):
    return tt.mean(tt.sum((actual-target)**2, axis = 1), axis = 0)


@symbolic_stateless
def percent_correct(actual, target):
    return tt.mean(tt.eq(tt.argmax(actual, axis=1), target), axis = 0) * 100
