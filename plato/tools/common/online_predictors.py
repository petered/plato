from abc import ABCMeta, abstractmethod
from plato.interfaces.decorators import symbolic_simple, symbolic_updater
from plato.interfaces.interfaces import IParameterized
from plato.tools.optimization.cost import get_named_cost_function
from artemis.ml.predictors.i_predictor import IPredictor


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

    def __init__(self, function, cost_function, optimizer, assert_all_params_optimized = True, regularization_cost = None):
        """
        :param function: Can be:
            A symbolic_simple function and an IParameterized object
            A subclass of FeedForwardModule, in which case it can implement train_call and test call separately
        :param cost_function: A symbolic function of the form :
            cost = cost_function(output, target)
            Where cost is a scalar, output is an (n_samples, ...) array representing the output of the function, and
            target is an (n_samples, ...) array representing the labels.
        :param optimizer: Is an IGradientOptimizer object (it takes a list of parameters and gradients and returns updates)
            OR, you can also have a dict<IGradientOptimizer: list<SharedVariable>>, which allows you to specify different
            optimizers for different parameters in the model.
        :param: assert_all_params_optimized: Only IF you specify optimizer as a dict, assert that all parameters are
            being optimzized.
        :param regularization_cost: Optionally, a function of the form:
            cost = regularization_cost(params)
            Where cost is a scalar and params is the list of shared variables returned by function.parameters
        """
        self._function = function
        if isinstance(cost_function, str):
            cost_function = get_named_cost_function(cost_function)
        self._cost_function = cost_function
        self._regularization_cost = regularization_cost
        self._optimizer = optimizer
        self.assert_all_params_optimized = assert_all_params_optimized

    @symbolic_simple
    def predict(self, inputs):
        return self._function.test_call(inputs) if isinstance(self._function, FeedForwardModule) else self._function(inputs)

    @symbolic_updater
    def train(self, inputs, labels):
        feedforward_module = self._function if isinstance(self._function, FeedForwardModule) else ParametrizedFeedForwardModule(self._function)
        feedforward_module.train(x=inputs, y=labels, optimizer=self._optimizer, assert_all_params_optimized = self.assert_all_params_optimized, cost_fcn=self._cost_function, regularization_cost=self._regularization_cost)

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


class FeedForwardModule(IParameterized):

    def train_call(self, x):
        return self.__call__(x)

    def test_call(self, x):
        return self.__call__(x)

    def __call__(self, x):
        """
        :param x: Input tensor
        :returns: Another tensor
        """
        raise NotImplementedError()

    def train(self, x, y, cost_fcn, optimizer, assert_all_params_optimized=False, regularization_cost = None):
        cost = cost_fcn(self.train_call(x), y)
        if regularization_cost is not None:
            cost = cost + regularization_cost(self.parameters)
        if isinstance(optimizer, dict):
            # In this secret option, you can specify an dict with optimizers as keys and lists of parameters as values.
            catch_all_optimizer = None
            for suboptimizer, param_list in optimizer.iteritems():
                if param_list=='remaining':
                    assert catch_all_optimizer is None, "You cannot have more than one optimizer set to optimize the 'remaining' parameters"
                    catch_all_optimizer = suboptimizer
                else:
                    suboptimizer.update_parameters(cost=cost, parameters=param_list)
            optimized_params = [p for plist in optimizer.values() for p in plist]
            non_optimized_params = [p for p in self.parameters if p not in optimized_params]
            if catch_all_optimizer is not None:
                catch_all_optimizer.update_parameters(cost=cost, parameters=non_optimized_params)
            elif assert_all_params_optimized:
                optimized_params = [p for plist in optimizer.values() for p in plist]
                non_optimized_params = [p for p in self.parameters if p not in optimized_params]
                assert len(non_optimized_params)==0, "You specified assert_all_params_optimized=True, but did not include the following parameters of your model: {}".format(non_optimized_params)
        else:
            optimizer.update_parameters(cost=cost, parameters=self.parameters)

    @property
    def parameters(self):
        return []

    def to_spec(self):
        raise NotImplementedError("Need to specify")


class ParametrizedFeedForwardModule(FeedForwardModule):

    def __init__(self, parametrized_function):
        assert callable(parametrized_function) and hasattr(parametrized_function, 'parameters')
        self.parametrized_function = parametrized_function

    def __call__(self, x):
        return self.parametrized_function(x)

    @property
    def parameters(self):
        return self.parametrized_function.parameters
