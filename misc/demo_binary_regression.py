from itertools import count
import numpy as np
from matplotlib import pyplot as plt


__author__ = 'peter'


"""
Here, we do logistic regression with binary weights.
"""

def get_binary_regression_dataset(
        n_dims = 10,
        noise = 0.1,
        n_training = 1000,
        n_test = 100,
        ):
    n_samples = n_training+n_test
    x = (np.random.rand(n_samples, n_dims) > 0.5)*2-1
    w = np.random.rand(n_dims, 1)
    z = sigm(x.dot(w))
    y = (z > np.random.rand(*z.shape)).astype(int)
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
    # w = (np.random.rand(n_dim_in, 1) > 0.5).astype('int')
    w = np.zeros((n_dim_in, n_dim_out), dtype = 'int')
    rng = np.random.RandomState(seed)

    for i in count():
        # Horribly inefficient, but nothing compared to the python loop iteration.   If we want to do it fast,
        # we'll do it in theano.
        ix = i % n_dim_in
        # w_old = w[ix, 0]
        w[ix, :] = 0
        z_0 = sigm(x.dot(w))  # (n_samples, n_dim_out)
        w[ix, :] = 1
        z_1 = sigm(x.dot(w))  # (n_samples, n_dim_out)
        p_wa = sigm(np.sum(np.log(bernoulli(y, z_0))-np.log(bernoulli(y, z_1)), axis = 0))  # (n_dim_out, )
        # print p_wa
        w[ix, :] = p_wa > rng.rand(n_dim_out)
        # print '%s-->%s' % (w_old, w[ix, :])
        # if w[ix, 0] != w_old:
        #     print 'Switch from %s to %s at step %s' % (w_old, w[ix, 0], i)

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

    for i in count():
        ix = i % n_dim_in
        w_old = w[ix, 0]
        w[ix, :] = 0
        z_0 = sigm(x.dot(w))  # (n_samples, n_dim_out)
        w[ix, :] = 1
        z_1 = sigm(x.dot(w))  # (n_samples, n_dim_out)
        z=np.array([z_0, z_1])  # (2, n_samples, n_dim_out)
        p = bernoulli(y, z)  # (2, n_samples, n_dim_out)
        log_prod_p = np.sum(np.log(p), axis = 1)  # (2, n_dim_out)
        log_prod_p = log_prod_p-np.max(log_prod_p)   # (2, n_dim_out)
        prod_p = np.exp(log_prod_p)
        # print prod_p
        norm_prod_p_1 = prod_p[1]/np.sum(prod_p, axis = 0)  # (n_dim_out, )
        # log_prod_p_1 = log_prod_p[1, :],  # (n_dim_out, )
        w[ix, :] = norm_prod_p_1 > np.random.rand(n_dim_out)
        # if w[ix, 0] != w_old:
        #     print 'Switch from %s to %s at step %s' % (w_old, w[ix, 0], i)
        yield w.copy()


def demo_binary_regression():

    n_steps = 100000

    # First, lets make a dataset which could potentially be solved by this approach.
    x_tr, x_ts, y_tr, y_ts, w_true = get_binary_regression_dataset()

    # Now, to do this with Gibbs, we need    some
    sampler = binary_logistic_regression_sampler(x=x_tr, y=y_tr)
    w_mats = [w for _, w in zip(xrange(n_steps), sampler)]
    cumulative_test_error = mean_squared_error(y_ts[None], cummean([sigm(x_ts.dot(w)) for w in w_mats], axis = 0))
    mean_true_error = mean_squared_error(y_ts, sigm(x_ts.dot(w_true)))

    plt.semilogx(np.arange(1, len(cumulative_test_error)+1), cumulative_test_error, 'r')
    plt.axhline(mean_true_error)
    plt.xlabel('Iteration')
    plt.ylabel('Mean Squared Error')
    plt.legend(['Gibbs', 'Optimal'], loc = 'best')
    # plt.set_xscale('log')
    plt.show()


if __name__ == '__main__':

    demo_binary_regression()
