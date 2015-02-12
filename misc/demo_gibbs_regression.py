from collections import OrderedDict
import inspect
from utils.datasets.synthetic_linear import get_synthethic_linear_dataset
import numpy as np
import matplotlib.pyplot as pp

__author__ = 'peter'

"""
In these examples, we do univariate and multivariate regression using gibbs sampling.
"""


def demo_univariate_regression():

    print '%s\nUnivariate Regression Demo' % ('='*20, )

    # Get univariate regression data
    beta_0_true = 3
    beta_1_true = 1.4
    sigma_true = 0.8
    x = np.random.randn(120)
    y = beta_0_true + beta_1_true * x + sigma_true*np.random.randn(120)
    x_tr, x_ts = x[:100], x[100:]
    y_tr, y_ts = y[:100], y[100:]

    # Do analytical solution
    beta_1_est = (np.cov(x_tr, y_tr) / np.var(x_tr))[0, 1]
    beta_0_est = np.mean(y_tr) - beta_1_est * np.mean(x_tr)
    mean_squared_error = lambda actual, target: np.mean((actual-target)**2, axis = 0)
    analytical_solution_error = mean_squared_error(beta_0_est+beta_1_est*x_ts, y_ts)
    print 'Analytical solution cost: %s' % analytical_solution_error

    # Do gibbs solution
    n_steps = 10
    sample_normal = lambda mean, var: np.random.randn()*np.sqrt(var) + mean
    sampler = GibbsSampler(
        initial_values = {
            'beta_0': 0,
            'beta_1': 0,
            'sigma_sq': 1,
            'x': x_tr,
            'y': y_tr,
            'n': len(x_tr),
            'eps': 0.001
            },
        updates = OrderedDict([
            ('beta_0', lambda beta_1, sigma_sq, x, y, n: sample_normal(mean = np.sum(y-beta_1*x)/n, var = sigma_sq/n)),
            ('beta_1', lambda beta_0, x, y, sigma_sq: sample_normal(mean = (np.dot(x, y) - beta_0*np.sum(x))/np.sum(x**2), var = sigma_sq/np.sum(x**2))),
            ('sigma_sq', lambda eps, n, x, y, beta_0, beta_1: 1./np.random.gamma(eps+n/2., eps+np.sum((y-beta_0-beta_1*x)**2)/2)),
            ])
        )
    values = sampler(0)
    value_history = []
    for i, x in enumerate(xrange(n_steps)):
        gibbs_error = mean_squared_error(values['beta_0']+values['beta_1']*x_ts, y_ts)
        print 'Gibbs cost at step %s: %s' % (i, gibbs_error)
        values = sampler(1)
        value_history.append(values.copy())

    # Plot Results:
    pp.plot(x_tr, y_tr, 'b.')
    pp.plot(x_ts, y_ts, 'bx')
    x_pts = np.linspace(-5, 5, 10)
    pp.plot(x_pts, beta_0_true+beta_1_true*x_pts, 'b')
    pp.plot(x_pts, beta_0_est+beta_1_est*x_pts, 'r')
    for i, v in enumerate(value_history):
        frac = float(i)/n_steps
        pp.plot(x_pts, v['beta_0'] + v['beta_1']*x_pts, color = tuple(np.array([.7, 1, .7])*(1-frac) + np.array([0, .7, 0])*frac))
    pp.gca().xaxis.grid(True)
    pp.gca().yaxis.grid(True)
    pp.legend(['Training Points', 'Test Points', 'True Solution', 'Analytical Solution', 'Gibbs Solutions'], loc = 'best')
    pp.show()


def demo_multivariate_regression():
    """
    In this demo we implement linear regression with gibbs sampling and compare the result to plain old linear regression.

    This is a terrible use-case for gibbs sampling, but a nice way to see that it's working.

    :return:
    """

    print '%s\nMultivariate Regression Demo' % ('='*20, )

    # Get linear regression dataset (20-D inputs, 1-D output, 1000 samples training, 1000 samples test)
    dataset = get_synthethic_linear_dataset(n_output_dims = 1)
    x_tr = dataset.training_set.input
    y_tr = dataset.training_set.target
    x_ts = dataset.test_set.input
    y_ts = dataset.test_set.target
    cost_function = lambda actual, target: np.mean(np.sum((actual-target)**2, axis = 1), axis = 0)

    # Do random_matrix
    random_w = np.random.randn(x_tr.shape[1], y_tr.shape[1])
    random_prediction = np.dot(x_ts, random_w)
    random_cost = cost_function(random_prediction, y_ts)
    print 'Random cost: %s' % random_cost

    # Do standard Least Squares
    least_squares_w = np.linalg.solve(np.dot(x_tr.T, x_tr), np.dot(x_tr.T, y_tr))
    least_squares_prediction = np.dot(x_ts, least_squares_w)
    least_squares_cost = cost_function(least_squares_prediction, y_ts)
    print 'Analytical Solution cost: %s' % least_squares_cost

    # Find parameters by Gibbs sampling.
    # Following this tutorial: http://faculty.agecon.vt.edu/moeltner/AAEC5126/module5/BayesNormalIndependent.pdf
    # Note: There are only 2 random variables (beta_sigma_sq) in this example - the others are deterministic updates of
    # the parameters.
    sampler = GibbsSampler(
        initial_values={
            'mu': np.zeros((x_tr.shape[1], 1)),
            'v': np.diag(np.ones(x_tr.shape[1])),
            'nu': 1.,
            'tau': 1.,
            'x': x_tr,
            'y': y_tr,
            'n': x_tr.shape[1],
            },
        updates = OrderedDict([
            ('beta', lambda mu, v: np.random.multivariate_normal(mu[:, 0], v)[:, None]),
            ('sigma_sq', lambda nu, tau: np.array(1.)/np.random.gamma(shape = nu, scale = tau)),
            ('v_old', lambda v: v),
            ('v', lambda v, sigma_sq, x: np.linalg.inv(np.linalg.inv(v)+(1./sigma_sq)*np.dot(x.T, x))),
            ('mu', lambda mu, v, v_old, nu, y, x, beta, sigma_sq: np.dot(v, (np.linalg.solve(v_old, mu) + (1./sigma_sq)*np.dot(x.T, y)))),
            ('nu', lambda nu, n: (2*nu + n)/2),
            ('tau', lambda tau, y, x, beta: (2*tau + np.dot((y-np.dot(x, beta)).T, y-np.dot(x, beta)))/2),
            ])
        )

    for i, x in enumerate(xrange(10)):
        values = sampler(1)
        gibbs_prediction = np.dot(x_ts, values['beta'])
        gibbs_error = cost_function(gibbs_prediction, y_ts)
        print 'Gibbs cost at step %s: %s' % (i, gibbs_error)


class GibbsSampler(object):

    def __init__(self, initial_values, updates):
        """
        :param initial_values: A dict mapping variable name to initial value
        :param updates: An OrderedDict mapping variable name to update function.
        """
        assert isinstance(initial_values, dict)
        assert isinstance(updates, OrderedDict)
        self._values = initial_values
        self._updates = updates
        update_args = {}
        assigned_variables = set(self._values)
        for k, f in updates.iteritems():
            args = inspect.getargspec(f)[0]
            for a in args:
                assert a in assigned_variables, 'Variable "%s" is used before it is assigned (in the update function for %s)' % (a, k)
            assigned_variables.add(k)
            update_args[k] = args
        self._update_argument_pairs = OrderedDict((var_name, (updates[var_name], update_args[var_name])) for var_name in updates)

    def __call__(self, n_steps):
        for i in xrange(n_steps):
            for var_name, (var_update, update_args) in self._update_argument_pairs.iteritems():
                self._values[var_name] = var_update(**{k: self._values[k] for k in update_args})
        return self._values

if __name__ == '__main__':

    demo_univariate_regression()
    demo_multivariate_regression()
