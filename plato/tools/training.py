__author__ = 'peter'


class SupervisedTrainingFunction(object):

    def __init__(self, classifier, cost_function, optimizer):
        self._classifier = classifier
        self._cost_function = cost_function
        self._optimizer = optimizer

    def __call__(self, data, labels):
        output = self._classifier(data)
        cost = self._cost_function(output, labels)
        updates = self._optimizer(self._classifier.parameters, cost)
        return updates


class SupervisedTestFunction(object):

    def __init__(self, classifier, cost_function):
        self._classifier = classifier
        self._cost_function = cost_function

    def __call__(self, data, labels):
        output = self._classifier(data)
        cost = self._cost_function(output, labels)
        return cost
