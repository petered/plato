from general.mymath import softmax
from plato.tools.cost import percent_correct, mean_squared_error, negative_log_likelihood, \
    softmax_negative_log_likelihood, normalized_negative_log_likelihood
import numpy as np
from utils.tools.processors import OneHotEncoding
import pytest

__author__ = 'peter'


def test_cost_functions():

    actual = np.random.rand(100, 3)
    target = np.random.randint(3, size = 100)
    onehot_target = OneHotEncoding(n_classes=3)(target)

    assert np.allclose(percent_correct.compile()(actual, target), 100*(np.argmax(actual, axis=1)==target).mean())
    assert np.allclose(mean_squared_error.compile()(actual, onehot_target), (((actual-onehot_target)**2).sum(axis=1)).mean())

    nll_func = negative_log_likelihood.compile()

    with pytest.raises(AssertionError):
        nll_func(actual, target)

    softmax_actual = softmax(actual, axis=1)

    assert np.allclose(nll_func(softmax_actual, target),
        -np.log(softmax_actual[np.arange(actual.shape[0]), target]).mean())

    assert np.allclose(softmax_negative_log_likelihood.compile()(actual, target),
           nll_func(softmax_actual, target))

    normalized_actual = actual/actual.sum(axis=1, keepdims=True)

    assert np.allclose(normalized_negative_log_likelihood.compile()(normalized_actual, target),
           nll_func(normalized_actual, target))


if __name__ == '__main__':
    test_cost_functions()
