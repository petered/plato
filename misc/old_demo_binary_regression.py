from itertools import count
import numpy as np
from matplotlib import pyplot as plt

__author__ = 'peter'

"""
Here, we do logistic regression with binary weights.

This script has been replaced by a more up-to-date version, which uses our framework for testing predictors.  However,
we keep this script because it's nice and stand-alone - useful as an example.
"""


def get_binary_regression_dataset(
        n_dims = 20,
        n_training = 10000,
        n_test = 1000,
        noise_factor = 1,
        seed = 5354355
        ):
    rng = np.random.RandomState(seed)
    n_samples = n_training+n_test
    # Since magnitude of x.dot(w) grows with square-root of len(x), we scale x depending on n_dims
    x_scale = 1./(noise_factor*np.sqrt(n_dims))
    x = ((rng.rand(n_samples, n_dims) > 0.5)*2-1)*x_scale
    w = rng.rand(n_dims, 1)
    z = sigm(x.dot(w))
    y = (z > rng.rand(*z.shape)).astype(int)
    return x[:n_training], x[n_training:], y[:n_training], y[n_training:], w


sigm = lambda x: 1./(1+np.exp(-x))
bernoulli = lambda k, p: (p**k)*((1-p)**(1-k))  # Maybe a not the fastest way to do it but whatevs
mean_squared_error = lambda actual, target: np.mean(np.sum((actual-target)**2, axis = -1), axis = -1)


def cummean(x, axis):
    x=np.array(x)
    normalized = np.arange(1, x.shape[axis]+1).astype(float)[(slice(None), )+(None, )*(x.ndim-axis-1)]
    return np.cumsum(x, axis)/normalized


def binary_logistic_regression_sampler(
        x,
        y,
        seed = None,
        ):

    n_dim_in = x.shape[1]
    n_dim_out = y.shape[1]
    w = np.zeros((n_dim_in, n_dim_out), dtype = 'int')
    rng = np.random.RandomState(seed)

    for i in count():
        # Horribly inefficient, but nothing compared to the python loop iteration.   If we want to do it fast,
        # we'll do it in theano.
        ix = i % n_dim_in
        w[ix, :] = 0
        z_0 = sigm(x.dot(w))  # (n_samples, n_dim_out)
        w[ix, :] = 1
        z_1 = sigm(x.dot(w))  # (n_samples, n_dim_out)
        p_wa = sigm(np.sum(np.log(bernoulli(y, z_1))-np.log(bernoulli(y, z_0)), axis = 0))  # (n_dim_out, )
        w[ix, :] = p_wa > rng.rand(n_dim_out)
        yield w.copy()


def other_binary_logistic_regression_sampler(
        x,
        y,
        seed = None
        ):
    """
    Peter's method (equivalent)?
    :param x:
    :param y:
    :param seed:
    :return:
    """
    n_dim_in = x.shape[1]
    n_dim_out = y.shape[1]
    w = np.zeros((n_dim_in, n_dim_out), dtype = 'int')
    rng = np.random.RandomState(seed)

    for i in count():
        ix = i % n_dim_in
        w[ix, :] = 0
        z_0 = sigm(x.dot(w))  # (n_samples, n_dim_out)
        w[ix, :] = 1
        z_1 = sigm(x.dot(w))  # (n_samples, n_dim_out)
        z=np.array([z_0, z_1])  # (2, n_samples, n_dim_out)
        p = bernoulli(y, z)  # (2, n_samples, n_dim_out)
        log_prod_p = np.sum(np.log(p), axis = 1)  # (2, n_dim_out)
        log_prod_p = log_prod_p-np.max(log_prod_p)   # (2, n_dim_out)
        prod_p = np.exp(log_prod_p)
        norm_prod_p_1 = prod_p[1]/np.sum(prod_p, axis = 0)  # (n_dim_out, )
        w[ix, :] = norm_prod_p_1 > rng.rand(n_dim_out)
        yield w.copy()


def demo_binary_regression(which_sampler = 'max', noise_level = 0.1):

    n_steps = 10000

    # First, lets make a dataset which could potentially be solved by this approach.
    x_tr, x_ts, y_tr, y_ts, w_true = get_binary_regression_dataset(noise_factor=noise_level)

    # Get the sampler.  2 choices - the one max had on the board, and peter's version.  I think they're equivalent.
    sampler = binary_logistic_regression_sampler(x=x_tr, y=y_tr) \
        if which_sampler == 'max' else \
        other_binary_logistic_regression_sampler(x=x_tr, y=y_tr)

    # Accumulate the list of matrices and figure out the error
    print 'Training...'
    w_mats = [w for _, w in zip(xrange(n_steps), sampler)]
    print '...Done.  Testing...'
    cumulative_test_error = mean_squared_error(y_ts[None], cummean([sigm(x_ts.dot(w)) for w in w_mats], axis = 0))
    cumulative_training_error = mean_squared_error(y_tr[None], cummean([sigm(x_tr.dot(w)) for w in w_mats], axis = 0))
    true_test_error = mean_squared_error(y_ts, sigm(x_ts.dot(w_true)))
    true_training_error = mean_squared_error(y_tr, sigm(x_tr.dot(w_true)))
    print 'Done.'

    # Plot the results.
    plt.semilogx(np.arange(1, len(cumulative_training_error)+1), cumulative_training_error, 'r--')
    plt.semilogx(np.arange(1, len(cumulative_test_error)+1), cumulative_test_error, 'r')
    plt.axhline(true_training_error, color='b', linestyle = '--')
    plt.axhline(true_test_error, color='b', linestyle = '-')
    plt.xlabel('Iteration')
    plt.ylabel('Mean Squared Error')
    plt.legend(['Gibbs-Training Error', 'Gibbs-Test Error', 'Optimal-Training', 'Optimal-Test'], loc = 'best')
    plt.show()


if __name__ == '__main__':

    which_sampler = 'max'
    noise_level = 0.3

    demo_binary_regression(which_sampler = which_sampler, noise_level=noise_level)
