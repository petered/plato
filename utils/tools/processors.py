import numpy as np

__author__ = 'peter'


class OneHotEncoding(object):

    def __init__(self, n_classes = None, form = 'bin', dtype = None):
        assert form in ('bin', 'sign')
        if dtype is None:
            dtype = np.int32 if form == 'sign' else bool
        self._n_classes = n_classes
        self._dtype = dtype
        self.form = form

    def __call__(self, data):
        if self._n_classes is None:
            self._n_classes = np.max(data)+1
        out = np.zeros((data.size, self._n_classes, ), dtype = self._dtype)
        if self.form == 'sign':
            out[:] = -1
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


class RunningAverageWithBurnin(object):

    def __init__(self, burn_in_steps):
        self._burn_in_step_remaining = burn_in_steps
        self.averager = RunningAverage()

    def __call__(self, x):

        if self._burn_in_step_remaining > 0:
            self._burn_in_step_remaining-=1
            return x
        else:
            return self.averager(x)
