import numpy as np
from artemis.ml.tools.neuralnets import initialize_network_params
from plato.tools.common.online_predictors import GradientBasedPredictor
from plato.tools.mlp.manual_backprop_net import ManualBackpropNet, ExactBackpropLayer
from plato.tools.mlp.mlp import MultiLayerPerceptron
from plato.tools.optimization.optimizers import GradientDescent


def test_exact_manual_backprop_net():

    rng = np.random.RandomState(1234)

    n_samples = 5
    n_in, n_hid1, n_hid2, n_out = 10, 8, 7, 6
    ws = initialize_network_params(layer_sizes=[n_in, n_hid1, n_hid2, n_out], include_biases=False)
    x, y = rng.randn(n_samples, n_in), rng.randn(n_samples, n_out)

    auto_mlp = GradientBasedPredictor(
        function = MultiLayerPerceptron.from_weights(weights=ws, hidden_activations='relu', output_activation='linear'),
        cost_function='softmax-xe',
        optimizer=GradientDescent(0.1)
        )
    stick_mlp = ManualBackpropNet(
        layers = [ExactBackpropLayer(ws[0], 'relu'), ExactBackpropLayer(ws[1], 'relu'), ExactBackpropLayer(ws[2], 'linear')],
        optimizer = GradientDescent(0.1),
        loss = 'softmax-xe'
        )
    stick_shifted_by_robot = ManualBackpropNet(
        layers = MultiLayerPerceptron.from_weights(weights=ws, hidden_activations='relu', output_activation='linear').layers,
        optimizer = GradientDescent(0.1),
        loss = 'softmax-xe'
        )

    # Check forward passes match
    fp_auto = auto_mlp.predict.compile()
    fp_stick = stick_mlp.predict.compile()
    fp_robot = stick_shifted_by_robot.predict.compile()

    out_auto = fp_auto(x)
    out_stick = fp_stick(x)
    out_robot = fp_robot(x)
    assert np.allclose(out_auto, out_stick)
    assert np.allclose(out_auto, out_robot)

    # 1 Iteration of training
    ft_auto = auto_mlp.train.compile()
    ft_stick = stick_mlp.train.compile()
    ft_robot = stick_shifted_by_robot.train.compile()
    ft_auto(x, y)
    ft_stick(x, y)
    ft_robot(x, y)

    # Check parameter changes match
    dw0_auto = auto_mlp._function.layers[0].linear_transform.w.get_value() - ws[0]
    dw0_stick = stick_mlp.model.layers[0].linear_transform.w.get_value() - ws[0]
    dw0_robot = stick_shifted_by_robot.model.layers[0].linear_transform.w.get_value() - ws[0]
    assert np.allclose(dw0_auto, dw0_stick)
    assert np.allclose(dw0_auto, dw0_robot)

    # Check outputs match
    new_out_auto = fp_auto(x)
    new_out_stick = fp_stick(x)
    new_out_robot = fp_robot(x)
    assert np.allclose(new_out_auto, new_out_stick)
    assert not np.allclose(new_out_stick, out_auto)
    assert np.allclose(new_out_auto, new_out_robot)


if __name__ == '__main__':
    test_exact_manual_backprop_net()
