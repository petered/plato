from abc import ABCMeta, abstractmethod
from plato.core import symbolic, symbolic_single_output_updater
from plato.interfaces.decorators import symbolic_simple, symbolic_updater
from plato.interfaces.interfaces import IParameterized
from plato.tools.optimization.cost import get_named_cost_function
from utils.predictors.i_predictor import IPredictor

__author__ = 'peter'


class ISymbolicPredictor(object):
    """
    Online online_prediction have an initial state, and learn iteratively through repeated calls to train.
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    @symbolic_simple
    def predict(self, inputs):
        """
        :param inputs: A (n_samples, ...) tensor of inputs
        :return: outputs: A (n_samples, ...) tensor representing the classifier ouput
        """
        pass


    @abstractmethod
    @symbolic_updater
    def train(self, inputs, labels):
        """
        :param inputs: A (n_samples, ...) tensor of inputs
        :param labels: A (n_samples, ...) tensor of labels
        :return: updates: A list of 2-tuples representing parameter updates.
        """

    def compile(self, **kwargs):
        """
        Compile the predict and train methods to create an IPredictor object, which can take
        numerical (as opposed to symbolic) data. 

        see: utils.predictors.IPredictor
        """
        return CompiledSymbolicPredictor(self, **kwargs)


class GradientBasedPredictor(ISymbolicPredictor, IParameterized):

    def __init__(self, function, cost_function, optimizer):
        """
        :param function: Is a symbolic_simple function and an IParameterized object
        :param cost_function: Is an ICostFunction
        :param optimizer: Is an IGradientOptimizer
        """
        self._function = function
        if isinstance(cost_function, str):
            cost_function = get_named_cost_function(cost_function)
        self._cost_function = cost_function
        self._optimizer = optimizer

    @symbolic  # Allows stateful or non-stateful output
    def predict(self, inputs):
        return self._function(inputs)

    @symbolic_updater
    def train(self, inputs, labels):
        outputs, state_updates = self._function.to_format(symbolic_single_output_updater)(inputs)
        cost = self._cost_function(outputs, labels)
        updates = self._optimizer(cost = cost, parameters = self._function.parameters)
        return updates + state_updates

    @property
    def parameters(self):
        opt_params = self._optimizer.parameters if isinstance(self._optimizer, IParameterized) else []
        return self._function.parameters + opt_params


class CompiledSymbolicPredictor(IPredictor, IParameterized):
    """
    A Predictor containing the compiled methods for a SymbolicPredictor.
    """

    def __init__(self, symbolic_predictor, **kwargs):
        self.train_function = symbolic_predictor.train.compile(**kwargs)
        self.predict_function = symbolic_predictor.predict.compile(**kwargs)
        self._params = symbolic_predictor.parameters if isinstance(symbolic_predictor, IParameterized) else []
        self.symbolic_predictor=symbolic_predictor

    def train(self, input_data, target_data):
        self.train_function(input_data, target_data)

    def predict(self, input_data):
        return self.predict_function(input_data)

    @property
    def parameters(self):
        return self._params
