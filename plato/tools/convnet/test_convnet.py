import pickle
from collections import OrderedDict

import numpy as np
from artemis.plotting.db_plotting import dbplot

from artemis.general.mymath import argmaxnd
from artemis.ml.datasets.cifar import get_cifar_10_dataset
from artemis.ml.predictors.train_and_test import percent_argmax_correct
from plato.tools.common.online_predictors import GradientBasedPredictor
from plato.tools.common.training import assess_online_symbolic_predictor
from plato.tools.convnet.conv_specifiers import ConvInitSpec, NonlinearitySpec, PoolerSpec
from plato.tools.convnet.convnet import ConvNet, ConvLayer, Pooler, normalize_convnet, Nonlinearity, \
    ChannelwiseCrossCorr
from plato.tools.optimization.cost import negative_log_likelihood_dangerous
from plato.tools.optimization.optimizers import AdaMax

__author__ = 'peter'


def test_convnet_serialization():

    cifar10 = get_cifar_10_dataset(normalize_inputs=True, n_training_samples=50, n_test_samples=50)
    test_epochs = [0, 1, 2]
    assert cifar10.input_shape == (3, 32, 32)

    net = ConvNet.from_init(
        input_shape = cifar10.input_shape,
        w_init=0.01,
        specifiers=[
            ConvInitSpec(n_maps = 24, filter_size = (3, 3), mode = 'same'),
            NonlinearitySpec('relu'),
            PoolerSpec(region = 2, stride=2, mode='max'), # (16x16)
            ConvInitSpec(n_maps = 48, filter_size = (3, 3), mode = 'same'),
            NonlinearitySpec('relu'),
            PoolerSpec(region = 2, stride=2, mode='max'), # (8x8)
            ConvInitSpec(n_maps = 96, filter_size = (3, 3), mode = 'same'),
            NonlinearitySpec('relu'),
            PoolerSpec(region = 2, stride=2, mode='max'), # (4x4),
            ConvInitSpec(n_maps = 192, filter_size = (4, 4), mode = 'valid'), # (1x1)
            NonlinearitySpec('relu'),
            ConvInitSpec(n_maps = 10, filter_size = (1, 1), mode = 'valid'),
            NonlinearitySpec('softmax'),
            ],
        )

    predictor = GradientBasedPredictor(
        function = net,
        cost_function=negative_log_likelihood_dangerous,
        optimizer=AdaMax()
        )

    assess_online_symbolic_predictor(
        predictor = predictor,
        dataset = cifar10,
        evaluation_function=percent_argmax_correct,
        test_epochs=test_epochs,
        minibatch_size=20,
        add_test_values = False
        )

    results_1 = net.compile()(cifar10.test_set.input)

    savable = net.to_spec()
    serialized = pickle.dumps(savable)
    deserialized = pickle.loads(serialized)

    net_2 = ConvNet.from_init(deserialized,input_shape=cifar10.input_shape, rng=None)
    results_2 = net_2.compile()(cifar10.test_set.input)
    assert np.array_equal(results_1, results_2)


def test_normalize_convnet():

    rng = np.random.RandomState(1234)
    input_data = rng.randn(10, 3, 16, 16)

    net = ConvNet(layers=OrderedDict([
        ('conv_1', ConvLayer(w = rng.randn(5, 3, 3, 3), b = rng.randn(5, ), border_mode=1)),
        ('pool_1', Pooler((2, 2))),
        ('non_1', Nonlinearity('relu')),
        ('conv_2', ConvLayer(w = rng.randn(7, 5, 3, 3), b = rng.randn(7, ), border_mode=1)),
        ('pool_2', Pooler((2, 2))),
        ('non_2', Nonlinearity('relu')),
        ]))
    normalize_convnet(net, inputs = input_data)

    f = net.get_named_layer_activations.compile()

    act = f(input_data)

    for layer_name in ['conv_1', 'conv_2']:
        print '{layer}: {act}'.format(layer=layer_name, act=act[layer_name].std())
        assert 0.9999 < act[layer_name].std() < 1.0001


def test_cross_conv_layer():

    x_shift, y_shift = 3, -5
    rng = np.random.RandomState(1234)
    full_x = rng.randn(1, 10, 40, 40)
    x1 = full_x[:, :, 10:30, 10:30]
    x2 = full_x[:, :, 10+y_shift:30+y_shift, 10+x_shift:30+x_shift]
    func = ChannelwiseCrossCorr().compile()
    y = func((x1, x2))
    assert y.shape==(1, 10, 39, 39)
    dbplot(y)
    ixs = np.array([argmaxnd(y[0, i, :, :]) for i in xrange(10)])
    assert np.all(ixs-39//2 == (y_shift, x_shift))


if __name__ == '__main__':
    # test_convnet_serialization()
    # test_normalize_convnet()
    test_cross_conv_layer()
