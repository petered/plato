from abc import ABCMeta
from plato.interfaces.decorators import symbolic_stateless, symbolic_standard, symbolic_updater
from utils.predictors.i_predictor import IPredictor

__author__ = 'peter'


class IOnlinePredictor(object):
    """
    Online online_prediction have an initial state, and learn iteratively through repeated calls to train.
    """

    __metaclass__ = ABCMeta

    @symbolic_stateless
    def predict(self, inputs):
        """
        :param inputs: A (n_samples, ...) tensor of inputs
        :return: outputs: A (n_samples, ...) tensor representing the classifier ouput
        """
        pass


    @symbolic_updater
    def train(self, inputs, labels):
        """
        :param inputs: A (n_samples, ...) tensor of inputs
        :param labels: A (n_samples, ...) tensor of labels
        :return: updates: A list of 2-tuples representing parameter updates.
        """

    def compile(self, **kwargs):
        """
        Possible TODO, allow compilation of any class containing symbolic methods, which
        will result in a new class where all symbolic methods are compiled.
        :return:
        """
        return CompiledSymbolicPredictor(self, **kwargs)


class GradientBasedPredictor(IOnlinePredictor):

    def __init__(self, function, cost_function, optimizer):
        """
        :param function: Is a symbolic_stateless function and an IParameterized object
        :param cost_function: Is an ICostFunction
        :param optimizer: Is an IGradientOptimizer
        """
        self._function = function
        self._cost_function = cost_function
        self._optimizer = optimizer

    @symbolic_stateless
    def predict(self, inputs):
        return self._function(inputs)

    @symbolic_updater
    def train(self, inputs, labels):
        cost = self._cost_function(self._function(inputs), labels)
        updates = self._optimizer(cost = cost, parameters = self._function.parameters)
        return updates


class ISamplingPredictor(IOnlinePredictor):

    def update(self, x, y):
        pass

    def sample_posterior(self, x, y):
        pass


class CompiledSymbolicPredictor(IPredictor):

    def __init__(self, symbolic_predictor, **kwargs):
        self.train_function = symbolic_predictor.train.compile(**kwargs)
        self.predict_function = symbolic_predictor.predict.compile(**kwargs)

    def train(self, input_data, target_data):
        self.train_function(input_data, target_data)

    def predict(self, input_data):
        return self.predict_function(input_data)
