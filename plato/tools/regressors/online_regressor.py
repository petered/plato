from plato.interfaces.decorators import symbolic_updater, symbolic_stateless
from plato.interfaces.interfaces import IParameterized
from plato.tools.optimization.cost import negative_log_likelihood_dangerous, mean_xe, mean_squared_error
from plato.tools.common.online_predictors import ISymbolicPredictor
from plato.tools.optimization.optimizers import SimpleGradientDescent
import theano
import theano.tensor as tt
import numpy as np
__author__ = 'peter'


class OnlineRegressor(ISymbolicPredictor, IParameterized):
    """
    A Predictor that does online (logistic, multinomial, linear) regression.
    This can also be considered to be a one-layer neural network.

    For multi-layer extensions, see networks.py.
    """
    def __init__(self, input_size, output_size, regressor_type = 'multinomial', optimizer = SimpleGradientDescent(eta = 0.01),
            include_biases = True):

        self.w = theano.shared(np.zeros((input_size, output_size), dtype = theano.config.floatX))
        self.b = theano.shared(np.zeros(output_size, dtype = theano.config.floatX))
        self.optimizer = optimizer
        self.activation, self.cost_fcn = {
            'multinomial': (tt.nnet.softmax, negative_log_likelihood_dangerous),
            'logistic': (tt.nnet.sigmoid, mean_xe),
            'linear': (lambda x: x, mean_squared_error)
            }[regressor_type]
        self.include_biases = include_biases

    @symbolic_stateless
    def predict(self, x):
        """
        :param x: An (n_samples, input_size) array of inputs
        :return: An (n_samples, output_sze) array of output class probabilities
        """
        return self.activation(x.dot(self.w)+self.b)

    @symbolic_updater
    def train(self, x, targets):
        """
        :param x: An (n_samples, input_size) array of inputs
        :param targets: If this is a multinomial regressor, we expect an (n_samples, ) array of integer targets
            Otherwise (for logistic and linear), it's a (n_samples, output_size) array.
        :return: A list of parameter updates
        """
        y = self.predict(x)
        cost = self.cost_fcn(y, targets)
        updates = self.optimizer(cost = cost, parameters = self.parameters)
        return updates

    @property
    def parameters(self):
        return [self.w, self.b] if self.include_biases else [self.w]
