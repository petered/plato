import numpy as np

__author__ = 'peter'


class OneHotEncoding(object):

    def __init__(self, n_classes = None, dtype = bool):
        self._n_classes = n_classes
        self._dtype = dtype

    def __call__(self, data):
        if self._n_classes is None:
            self._n_classes = np.max(data)+1
        out = np.zeros((data.size, self._n_classes, ), dtype = self._dtype)
        if data.size > 0:  # Silly numpy
            out[np.arange(data.size), data.flatten()] = 1
        out = out.reshape(data.shape+(self._n_classes, ))
        return out

    def inverse(self, data):
        return np.argmax(data, axis = 1)


class RunningAverage(object):

    def __init__(self):
        self._n_samples_seen = 0
        self._average = 0

    def __call__(self, data):
        self._n_samples_seen+=1
        frac = 1./self._n_samples_seen
        self._average = (1-frac)*self._average + frac*data
        return self._average
