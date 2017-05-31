import symbol

import numpy as np
import pytest

from artemis.general.mymath import softmax
from plato.core import symbolic
from plato.tools.optimization.cost import percent_correct, mean_squared_error, negative_log_likelihood, \
    softmax_negative_log_likelihood, normalized_negative_log_likelihood, categorical_xe
from artemis.ml.tools.processors import OneHotEncoding
import theano.tensor as tt

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


def text_catxe_grads():

    rng = np.random.RandomState(1234)

    n_samples, n_dims = 5, 10
    z = rng.randn(n_samples, n_dims)
    yp = rng.rand(n_samples, n_dims)
    y = yp/yp.sum(axis=1, keepdims=True)
    should_be_grad = (-y + softmax(z, axis=1))/n_samples

    eps = 1e-7
    @symbolic
    def compute_loss(z_, y_):
        return categorical_xe(tt.nnet.softmax(z_), y_)
    f_loss = compute_loss.compile()

    @symbolic
    def compute_grad(z_, y_):
        return tt.grad(compute_loss(z_, y_), wrt=z_)
    f_grad = compute_grad.compile()

    grad_1 = f_grad(z, y)  # Check grad
    assert np.allclose(should_be_grad, grad_1)
    for i in xrange(n_samples):
        for j in xrange(n_dims):
            zp = z.copy()
            zp[i,j]+=eps
            emp_grad_ij = (f_loss(zp, y) - f_loss(z, y)) / eps
            assert np.allclose(emp_grad_ij, grad_1[i, j])


if __name__ == '__main__':
    # test_cost_functions()
    text_catxe_grads()