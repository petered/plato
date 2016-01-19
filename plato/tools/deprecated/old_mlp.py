import numpy as np
from plato.core import symbolic_simple
from plato.interfaces.interfaces import IParameterized
from plato.tools.mlp.mlp import Layer, FullyConnectedTransform

__author__ = 'peter'


@symbolic_simple
class OldMultiLayerPerceptron(IParameterized):
    """
    A Multi-Layer Perceptron.  This version is deprecated.  Use plato.tools.mlp.MultiLayerPerceptron, which has a
    new constructor format.
    """

    def __init__(self, layer_sizes, input_size, hidden_activation = 'sig', output_activation = 'sig',
            normalize_minibatch = False, scale_param = False, w_init = 0.1, use_bias = True):
        """
        :param layer_sizes: A list indicating the sizes of each layer.
        :param input_size: An integer indicating the size of the input layer
        :param hidden_activation: A string or list of strings indicating the type of each hidden layer.
            {'sig', 'tanh', 'rect-lin', 'lin', 'softmax'}
        :param output_activation: A string (see above) identifying the activation function for the output layer
        :param w_init: A function which, given input dims, output dims, return
        """
        if isinstance(w_init, (int, float)):
            val = w_init
            w_init = lambda n_in, n_out: val*np.random.randn(n_in, n_out)

        self.layers = [
            Layer(
                linear_transform = FullyConnectedTransform(
                    w = w_init(pre_size, post_size),
                    normalize_minibatch=normalize_minibatch,
                    scale = scale_param,
                    use_bias = use_bias
                    ),
                nonlinearity = nonlinearity
                )
            for pre_size, post_size, nonlinearity in zip(
                [input_size]+layer_sizes[:-1],
                layer_sizes,
                [hidden_activation] * (len(layer_sizes)-1) + [output_activation]
            )]

    def __call__(self, x):
        for lay in self.layers:
            x = lay(x)
        return x

    @property
    def parameters(self):
        return sum([l.parameters for l in self.layers], [])