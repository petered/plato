import time
from experimental.boltzmann_sampling import random_symmetric_matrix, gibbs_sample_boltzmann_py_naive, gibbs_sample_boltzmann_py_smart, \
    gibbs_sample_boltzmann_weave_naive, gibbs_sample_boltzmann_weave_smart
import numpy as np

__author__ = 'peter'


def profile_sampling_speed():
    """
    Gibbs sampling can be really slow in python because it can't just be turned into a big numpy vector operation,
    since each update of each unit depends on the last.  So here we experiment with different things to make it
    faster.
    1) Plain old python
    2) Scipy weave
    3) Theano scan op (Not done yet!)

    There are also two versions of each:
    naive computes the weights.dot(state) on every input
    smart just updates the dot product's result when states change

    So far, typical results look like this:
    ----
    Time for weave-smart: 0.0132260322571
    Time for python-smart: 1.98137593269
    Time for weave-naive: 0.0213210582733
    Time for python-naive: 1.14777588844
    ----
    Time for weave-smart (block): 0.0236368179321
    Time for python-smart (block): 0.150621175766
    Time for weave-naive (block): 0.0169339179993
    Time for python-naive (block): 0.158509969711
    ----
    So use weave.
    """

    n_dims = 20
    n_steps = 10000
    mag = 0.4
    weight_seed = 4231354325
    sampling_seed = 75432
    rng=np.random.RandomState(weight_seed)
    w = random_symmetric_matrix(n_dims=n_dims, mag = 0.4, rng=rng)
    b = mag*rng.randn(n_dims)

    sampling_functions = {
        'python-naive': gibbs_sample_boltzmann_py_naive,
        'python-smart': gibbs_sample_boltzmann_py_smart,
        'weave-naive': gibbs_sample_boltzmann_weave_naive,
        'weave-smart': gibbs_sample_boltzmann_weave_smart,
    }

    for block in [False, True]:

        results = {}
        for name, f in sampling_functions.iteritems():
            t_start = time.time()
            results[name] = f(w, b, n_steps = n_steps, rng = np.random.RandomState(sampling_seed), block = block)
            t_elapsed = time.time() - t_start
            print 'Time for %s%s: %s' % (name, ' (block)' if block else '', t_elapsed)

        print '----'
        for name, res in results.iteritems():
            assert np.allclose(res, results['python-naive']), '%s produced the wrong result' % name

    print ('All create close results')


if __name__ == '__main__':

    profile_sampling_speed()
