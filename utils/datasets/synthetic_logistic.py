from utils.datasets.datasets import DataSet, DataCollection
import numpy as np
from utils.tools.mymath import sigm

__author__ = 'peter'


def get_logistic_regression_dataset(**kwargs):
    x_tr, y_tr, x_ts, y_ts, _ = get_logistic_regression_data(**kwargs)
    return DataSet(DataCollection(x_tr, y_tr), DataCollection(x_ts, y_ts))


def get_logistic_regression_data(
        n_dims = 20,
        n_training = 1000,
        n_test = 100,
        noise_factor = 1,
        seed = 5354355
        ):
    rng = np.random.RandomState(seed)
    n_samples = n_training+n_test
    # Since magnitude of x.dot(w) grows with square-root of len(x), we scale x depending on n_dims
    x_scale = np.array(1.)/noise_factor  #/(noise_factor*np.sqrt(n_dims))
    x = ((rng.rand(n_samples, n_dims) > 0.5)*2-1)
    w = rng.rand(n_dims, 1)
    z = sigm(x.dot(w)*x_scale)
    y = (z > rng.rand(*z.shape)).astype(int)
    return x[:n_training], y[:n_training], x[n_training:], y[n_training:], w
