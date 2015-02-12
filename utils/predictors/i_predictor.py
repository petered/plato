from abc import abstractmethod

__author__ = 'peter'


class IPredictor(object):

    @abstractmethod
    def train(self, input_data, target_data):
        """
        :param input_data:
        :param target_data:
        :return:
        """

    @abstractmethod
    def predict(self, input_data):
        """
        :return: The output given the input data
        """
