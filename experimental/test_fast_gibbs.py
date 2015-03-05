import time
from experimental.boltzmann_sampling import generate_boltzmann_parameters, gibbs_sample_boltzmann_py_naive, gibbs_sample_boltzmann_py_smart, \
    gibbs_sample_boltzmann_weave_naive, gibbs_sample_boltzmann_weave_smart, herded_sample_boltzmann_py_naive, \
    herded_sample_boltzmann_weave_smart
import numpy as np
from plotting.easy_plotting import ezplot

__author__ = 'peter'


def profile_sampling_speed(test_mode = False):
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
    Time for gibbs-python-smart: 2.22626399994
    Time for gibbs-weave-naive: 0.0281620025635
    Time for gibbs-python-naive: 1.26503610611
    Time for gibbs-weave-smart: 0.0165710449219
    ----
    Time for gibbs-python-smart (block): 0.159940004349
    Time for gibbs-weave-naive (block): 0.0244989395142
    Time for gibbs-python-naive (block): 0.159762144089
    Time for gibbs-weave-smart (block): 0.0173401832581
    ----
    Time for herded-python-naive: 1.29901504517
    Time for herded-weave-smart: 0.0169270038605
    ----
    Time for herded-python-naive (block): 0.156569004059
    Time for herded-weave-smart (block): 0.0145919322968
    ----
    So use weave.
    """

    n_dims = 20
    n_steps = 400 if test_mode else 10000
    mag = 0.4
    weight_seed = 1234
    sampling_seed = 75434
    plot = False

    # TODO: Solve mystery of inexact results for herding.
    # We get a mysterious numerical error sometimes.  It does not appear to be due
    # to drift, because we still get it when we recomute the current on every iteration.
    # Set weight seed to 1235 to get the error
    rng=np.random.RandomState(weight_seed)
    w, b = generate_boltzmann_parameters(n_dims=n_dims, mag = 0.4, rng = np.random.RandomState(weight_seed))

    def compare_times_and_assert_correct(sampling_functions, trusted_sampler, block, kwarg_constructor=None):

        results = {}
        for name, f in sampling_functions.iteritems():
            kwargs = kwarg_constructor() if kwarg_constructor is not None else {}
            t_start = time.time()
            results[name] = f(w, b, n_steps = n_steps, block = block, **kwargs)
            t_elapsed = time.time() - t_start
            print 'Time for %s%s: %s' % (name, ' (block)' if block else '', t_elapsed)

        if plot:
            ezplot(results)

        print '----'

        for name, res in results.iteritems():
            assert np.allclose(res, results[trusted_sampler]), '%s produced the wrong result after %s iterations.' \
                % (name, np.argmax(np.sum(res!=results[trusted_sampler], axis = 1) > 0))

    gibbs_samplers = {
        'gibbs-python-naive': gibbs_sample_boltzmann_py_naive,
        'gibbs-python-smart': gibbs_sample_boltzmann_py_smart,
        'gibbs-weave-naive': gibbs_sample_boltzmann_weave_naive,
        'gibbs-weave-smart': gibbs_sample_boltzmann_weave_smart,
        }

    compare_times_and_assert_correct(gibbs_samplers, trusted_sampler = 'gibbs-weave-naive', block = False, kwarg_constructor = lambda: {'rng': np.random.RandomState(sampling_seed)})
    compare_times_and_assert_correct(gibbs_samplers, trusted_sampler = 'gibbs-weave-naive', block = True, kwarg_constructor = lambda: {'rng': np.random.RandomState(sampling_seed)})
    # Note: weave-smart block sampling currently just redirects to weave-naive block sampling
    # same with python-smart/python-naive

    herded_samplers = {
        'herded-python-naive': herded_sample_boltzmann_py_naive,
        'herded-weave-smart': herded_sample_boltzmann_weave_smart,
        }
    compare_times_and_assert_correct(herded_samplers, trusted_sampler = 'herded-python-naive', block = False)
    compare_times_and_assert_correct(herded_samplers, trusted_sampler = 'herded-python-naive', block = True)


def test_sampling_systems():

    profile_sampling_speed(test_mode = True)


if __name__ == '__main__':

    TEST_MODE = False

    if TEST_MODE:
        test_sampling_systems()
    else:
        profile_sampling_speed()
