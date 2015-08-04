from plato.tools.common.online_predictors import IPredictor
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


