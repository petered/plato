import numpy as np
from utils.datasets.synthethic_clusters import get_synthetic_clusters_dataset
from utils.datasets.synthetic_linear import get_synthethic_linear_dataset
from scipy.stats import norm

__author__ = 'peter'


def demo_probit_regression():

    x_tr, y_tr, x_ts, y_ts = get_synthetic_clusters_dataset(n_clusters = 2).xyxy
    y_tr = y_tr[:, None]
    y_ts = y_ts[:, None]

    prob_reg = probit_regressor(x=x_tr, y=y_tr, mu_b = np.zeros((x_tr.shape[1], 1)), sig_b = np.diag(np.ones(x_tr.shape[1])))
    results = [values for i, values in zip(xrange(100), prob_reg)]
    test_output = norm.cdf(x_ts.dot(results[-1]['b']))
    mse = lambda actual, target: np.mean((actual-target)**2)
    score = mse(test_output, y_ts)
    print 'Score: %s' % score


def probit_regressor(
        x,
        y,
        mu_b,
        sig_b,
        ):
    """
    A probit regressor implementing Gibbs Sampler I in this paper:
    http://www.cs.ubc.ca/~emtiyaz/Writings/EMTstatisticalcomputation.pdf

    (Does not work right now?  TODO: Why?)
    :param x:
    :param y:
    :param mu_b:
    :param sig_b:
    :return:
    """
    while True:
        b = np.random.multivariate_normal(mean = mu_b[:, 0], cov = sig_b)[:, None]
        z = (np.random.randn(*y.shape) + x.dot(b))
        z = z * np.equal(z>0, y)
        sig_b_old = sig_b
        sig_b = np.linalg.inv(np.linalg.inv(sig_b)+x.T.dot(x))
        mu_b = sig_b.dot(np.linalg.solve(sig_b_old, mu_b)+x.T.dot(z))
        yield locals()


if __name__ == '__main__':

    demo_probit_regression()
