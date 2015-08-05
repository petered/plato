from plato.interfaces.decorators import symbolic_updater, symbolic_simple

__author__ = 'peter'


@symbolic_updater
class SupervisedTrainingFunction(object):

    def __init__(self, classifier, cost_function, optimizer):
        self._classifier = classifier
        self._cost_function = cost_function
        self._optimizer = optimizer

    def __call__(self, data, labels):
        output = self._classifier(data)
        cost = self._cost_function(output, labels)
        updates = self._optimizer(cost=cost, parameters = self._classifier.parameters)
        return updates


@symbolic_simple
class SupervisedTestFunction(object):

    def __init__(self, classifier, cost_function):
        self._classifier = classifier
        self._cost_function = cost_function

    def __call__(self, data, labels):
        output = self._classifier(data)
        cost = self._cost_function(output, labels)
        return cost
