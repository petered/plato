import numpy as np

__author__ = 'peter'


class DataSet(object):

    def __init__(self, training_set, test_set, validation_set = None):
        self.training_set = training_set
        self.test_set = test_set
        self._validation_set = validation_set

    @property
    def validation_set(self):
        if self._validation_set is None:
            raise Exception('Validation set does not exist')
        else:
            return self._validation_set


class DataCollection(object):

    def __init__(self, inputs, targets):
        if isinstance(inputs, np.ndarray):
            inputs = (inputs, )
        if isinstance(targets, np.ndarray):
            targets = (targets, )
        n_samples = len(inputs[0])
        assert all(n_samples == len(d) for d in inputs) and all(n_samples = len(l) for l in targets)
        self._inputs = input
        self._targets = targets
        self._n_samples = n_samples

    @property
    def n_samples(self):
        return self._n_samples

    @property
    def input(self):
        only_input, = self._inputs
        return only_input

    @property
    def target(self):
        only_target, = self._targets
        return only_target

    def minibatch_iterator(self, minibatch_size = 1, epochs = 1, final_treatment = 'stop', single_channel = False):
        """
        :param minibatch_size:
        :param epochs:
        :param final_treatment:
        :param single_channel:
        :return: An iterator.  The iterator returns a tuple<tuple<*ndarray>, tuple<*ndarray>>, where the inner tuples
            are minibatches of the input and targets
        """
        i = 0
        total_samples = epochs * self._n_samples

        while i < total_samples:
            next_i = total_samples + minibatch_size
            segment = np.arange(i, next_i) % self._n_samples
            if next_i > total_samples:
                if final_treatment == 'stop':
                    break
                elif final_treatment == 'truncate':
                    next_i = total_samples
                else:
                    raise Exception('Unknown final treatment: %s' % final_treatment)
            if single_channel:
                input_minibatch, = [d[segment] for d in self._inputs]
                target_minibatch, = [d[segment] for d in self._targets]
            else:
                input_minibatch = self._input[segment]
                target_minibatch = self._target[segment]
            yield input_minibatch, target_minibatch
            i = next_i
