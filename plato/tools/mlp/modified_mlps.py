from plato.core import symbolic_simple
from plato.tools.mlp.mlp import MultiLayerPerceptron

__author__ = 'peter'


@symbolic_simple
class SequentialMultiLayerPerceptron(MultiLayerPerceptron):
    """
    This variant on a multi-layer perceptron processes samples in sequence rather than
    as a big batch operation.  The only time this will make a difference is when layers
    have some internal state that they want to update.
    """

    def __call__(self, x):
        output = self.process_sample.scan(sequences = [x[:, None, :]])
        return output[:, 0, :]

    @symbolic_simple
    def process_sample(self, x):
        assert x.ishape[0] == 1, "We expect x to have a minibatch of size 1."
        for lay in self.layers:
            x = lay(x)
        return x
