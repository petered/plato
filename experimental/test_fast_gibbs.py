import time
from experimental.boltzmann_sampling import random_symmetric_matrix, gibbs_sample_py_naive, gibbs_sample_py_smart, \
    gibbs_sample_weave_naive, gibbs_sample_weave_smart
import numpy as np
from plotting.easy_plotting import ezplot

__author__ = 'peter'


"""
Gibbs sampling can be really slow in python because it can't just be turned in a bit numpy vector operation,
since each update of each unit depends on the last.  So here we experiment with different thing to make it
faster.
1) Plain old python
2) Scipy weave
3) Theano scan op
"""


def test_fast_gibbs():

    n_dims = 20
    n_steps = 10000
    mag = 0.4
    weight_seed = 4231354325
    sampling_seed = 75432
    rng=np.random.RandomState(weight_seed)
    w = random_symmetric_matrix(n_dims=n_dims, mag = 0.4, rng=rng)
    b = mag*rng.randn(n_dims)

    sampling_functions = {
        'python-naive': gibbs_sample_py_naive,
        'python-smart': gibbs_sample_py_smart,
        'weave-naive': gibbs_sample_weave_naive,
        'weave-smart': gibbs_sample_weave_smart,
    }

    results = {}
    for name, f in sampling_functions.iteritems():
        t_start = time.time()
        results[name] = f(w, b, n_steps = n_steps, rng = np.random.RandomState(sampling_seed))
        t_elapsed = time.time() - t_start
        print 'Time for %s: %s' % (name, t_elapsed)

    for name, res in results.iteritems():
        assert np.allclose(res, results['python-naive']), '%s produced the wrong result' % name

    print ('All create close results')


if __name__ == '__main__':

    test_fast_gibbs()
