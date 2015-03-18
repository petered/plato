import numpy as np
from plotting.db_plotting import dbplot

__author__ = 'peter'


def test_dbplot():

    arr = np.random.rand(10, 10)

    for i in xrange(4):
        arr_sq=arr**2
        arr = arr_sq/np.mean(arr_sq)
        dbplot(arr, 'arr')
        for j in xrange(3):
            barr = np.random.randn(10, 2)
            dbplot(barr, 'barr')


if __name__ == '__main__':
    test_dbplot()
