from abc import ABCMeta, abstractmethod

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


class NegativeLogLikelihood(ICostFunction):

    def __call__(self, actual, target):
        """
        :param actual: An (n_samples, n_labels) tensor where rows are normalized and actual[i,j] indicates the belief
            that on sample[i] the correct target is j.  Note: it is ASSUMED that rows are probability distributions.
        :param target: An (n_samples, ) tensor indicating the target label for each sample
        :return: The average (over samples) of the negative log-likelihood.
        """
        raise NotImplementedError()


class PercentCorrect(ICostFunction):

    def __call__(self, actual, target):
        raise NotImplementedError()
