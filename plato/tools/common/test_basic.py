import numpy as np
from plato.tools.common.basic import running_variance, running_average, running_average


def test_running_stats():
    rng = np.random.RandomState(1234)
    var = 2.3
    mean = -1.4
    data = rng.randn(1000)*np.sqrt(var) + mean

    f = running_average.compile()
    averages = [f(d) for d in data]
    assert np.allclose(averages[-1], data.mean())
    assert -1.38 < averages[-1] < -1.37

    f = running_average.compile()
    averages = [f(d, decay=0.01) for d in data]
    assert -1.4 < np.mean(averages[-len(averages)/2:]) < -1.39

    f = running_variance.compile()
    variances = [f(d) for d in data]
    assert 2.177<variances[-1]<2.178
    assert np.allclose(variances[-1], np.var(data))

    f = running_variance.compile()
    variances = [f(d, decay=0.01) for d in data]
    assert 2.21 < np.mean(variances[-len(variances)/2:]) < 2.22

    # With initial value
    f = running_variance.partial(initial_value=1, shape=()).compile()
    variances = [f(d, decay=0.01) for d in data]
    assert 2.21 < np.mean(variances[-len(variances)/2:]) < 2.22
    assert 0.99 < variances[0] < 1.


def test_running_stats_aggregated():

    rng = np.random.RandomState(1234)
    var = 2.3
    mean = -1.4
    shape = (5, 3)
    data = rng.randn(1000, *shape)*np.sqrt(var) + mean

    # Running Mean
    f = running_average.partial(shape=shape).compile()  # TODO: Remove requirement of adding shape
    averages = [f(d) for d in data]
    assert np.allclose(averages[-1], data.mean(axis=0))

    # Running Aggregate mean
    f = running_average.partial(shape=shape, elementwise=False).compile()  #
    averages = [f(d) for d in data]
    assert np.allclose(averages[-1], data.mean())
    assert -1.38 < averages[-1] < -1.37

    # Running Aggregate variance
    f = running_variance.partial(shape=shape, elementwise=False).compile()  #
    variances = [f(d) for d in data]
    assert np.allclose(variances[-1], data.var())
    assert 2.28 < variances[-1] < 2.29

    # Decaying Running Aggregate Variacne
    f = running_variance.partial(shape=shape, elementwise=False).compile()  #
    variances = [f(d, decay=0.01) for d in data]
    assert 2.29 < np.mean(variances[-len(variances)/2:]) < 2.3
    # TODO: Verify that variance estimate is correct when elementwise=False
    # For example what if the channels have different means... still correct?

    # Decaying Running Aggregate Variacne with initialization
    f = running_variance.partial(shape=shape, elementwise=False, initial_value=1.).compile()  #
    variances = [f(d, decay=0.01) for d in data]
    assert 2.29 < np.mean(variances[-len(variances)/2:]) < 2.3
    assert 1.03 < variances[0] < 1.04

if __name__ == '__main__':
    # test_running_stats()
    test_running_stats_aggregated()
