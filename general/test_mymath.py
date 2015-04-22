from general.mymath import softmax, cummean, cumvar, sigm, expected_sigm_of_norm
import numpy as np
__author__ = 'peter'


def test_softmax():
    x = np.random.randn(3, 4, 5)

    s = softmax(x, axis=1)
    assert s.shape==(3, 4, 5) and (s>0).all() and (s<1).all() and np.allclose(np.sum(s, axis=1), 1)


def test_cummean():

    arr = np.random.randn(3, 4)
    cum_arr = cummean(arr, axis = 1)
    assert np.allclose(cum_arr[:, 0], arr[:, 0])
    assert np.allclose(cum_arr[:, 1], np.mean(arr[:, :2], axis = 1))
    assert np.allclose(cum_arr[:, 2], np.mean(arr[:, :3], axis = 1))


def test_cumvar():

    arr = np.random.randn(3, 4)
    cum_arr = cumvar(arr, axis = 1)
    assert np.allclose(cum_arr[:, 0], 0)
    assert np.allclose(cum_arr[:, 1], np.var(arr[:, :2], axis = 1))
    assert np.allclose(cum_arr[:, 2], np.var(arr[:, :3], axis = 1))


def test_exp_sig_of_norm():

    mean = 1
    std = 0.8
    n_points = 1000
    seed = 1234

    inputs = np.random.RandomState(seed).normal(mean, std, size = n_points)
    vals = sigm(inputs)
    sample_mean = np.mean(vals)

    for method in ('maclauren-2', 'maclauren-3', 'probit'):
        approx_true_mean = expected_sigm_of_norm(mean, std, method = method)
        approx_sample_mean = expected_sigm_of_norm(np.mean(inputs), np.std(inputs), method = method)
        true_error = np.abs(approx_true_mean-sample_mean)/sample_mean
        sample_error = np.abs(approx_sample_mean-sample_mean)/sample_mean
        print 'Error for %s: %.4f True, %.4f Sample.' % (method, true_error, sample_error)
        assert true_error < 0.02, 'Method %s did pretty bad' % (method, )


if __name__ == '__main__':

    test_exp_sig_of_norm()
    test_cumvar()
    test_cummean()
    test_softmax()
