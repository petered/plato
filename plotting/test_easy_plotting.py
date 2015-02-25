import numpy as np
from plotting.easy_plotting import ezplot
from matplotlib import pyplot as plt
__author__ = 'peter'


class DataContainer(object):

    def __init__(self, im, line, struct, text, number):
        self._im = im
        self._line = line
        self._struct = struct
        self._text = text
        self._number = number


def test_easy_plot():

    thing = DataContainer(
        im =np.random.randn(30, 40),
        line = np.sin(np.arange(100)/10.),
        struct = {'video': np.random.randn(17, 20, 30)},
        text = 'adsagfdsf',
        number = 5
        )
    plt.ion()
    ezplot(thing)


if __name__ == '__main__':
    test_easy_plot()
