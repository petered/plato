from utils.predictors.i_predictor import IPredictor

__author__ = 'peter'


class MockPredictor(IPredictor):

    def __init__(self, prediction_function):
        self._prediction_function = prediction_function

    def train(self, input_data, target_data):
        pass

    def predict(self, input_data):
        return self._prediction_function(input_data)
