from misc.demo_binary_regression import simple_binary_gibbs_regressor
import numpy as np
from scipy.stats.stats import ttest_ind
from utils.benchmarks.train_and_test import mean_squared_error
from utils.datasets.synthetic_logistic import get_logistic_regression_data
from utils.tools.progress_indicator import ProgressIndicator

__author__ = 'peter'


def test_binary_logistic_regression(plot = True):

    n_steps = 1000
    n_dims = 10
    noise_factor = 0.2

    x, y, _, _, _ = get_logistic_regression_data(n_training = 1000, n_test = 100, n_dims=n_dims, noise_factor = noise_factor)

    sampler = simple_binary_gibbs_regressor(n_dim_in = n_dims, n_dim_out = 1, seed = 5)

    train_fcn = sampler.update.compile(mode='tr')
    predict_fcn = sampler.sample_posterior.compile(mode = 'run')

    out = []
    pi = ProgressIndicator(n_steps, update_every=(1, 'seconds'))
    for i in xrange(n_steps):
        out.append(predict_fcn(x))
        train_fcn(x, y)
        pi()
    out = np.array(out)
    error = mean_squared_error(y, out)

    if plot:
        import matplotlib.pyplot as plt
        plt.semilogx(error)
        plt.show()

    t_stat, p_val = ttest_ind(error[:n_steps/2], error[n_steps/2:])

    assert t_stat>0 and p_val < 0.01


if __name__ == '__main__':

    test_binary_logistic_regression()
