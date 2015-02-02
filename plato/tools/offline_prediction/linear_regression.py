from plato.interfaces.decorators import symbolic_stateless, symbolic_standard
from plato.tools.online_prediction.online_predictors import IPredictor
import theano
import theano.tensor as ts

__author__ = 'peter'


class LinearRegression(IPredictor):
    """
    Offline linear regression
    """

    def predict(self, inputs, (w, )):
        return ts.dot(inputs, w)

    def train(self, inputs, targets):
        w = ts.dot(ts.dot(inputs.T, inputs), ts.dot(inputs.T, targets))
        return [w]


