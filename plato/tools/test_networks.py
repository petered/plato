from plato.interfaces.decorators import symbolic_updater
from plato.tools.cost import negative_log_likelihood_dangerous
from plato.tools.networks import MultiLayerPerceptron
from plato.tools.optimizers import SimpleGradientDescent
from utils.benchmarks.train_and_test import percent_argmax_correct
from utils.bureaucracy import zip_minibatch_iterate
from utils.datasets.synthetic_clusters import get_synthetic_clusters_dataset
import numpy as np

__author__ = 'peter'


def test_mlp(seed = 1234):
    """
    This verifies that the MLP works.  It's intentionally not using any wrappers on top of MLP to show its "bare bones"
    usage.  Wrapping in GradientBasedPredictor can simplify usage - see test_symbolic_predictors.
    """

    dataset = get_synthetic_clusters_dataset()

    rng = np.random.RandomState(seed)
    mlp = MultiLayerPerceptron(
        input_size = dataset.input_size,
        layer_sizes = [20, dataset.n_categories],
        hidden_activation = 'relu',
        output_activation = 'softmax',
        w_init = lambda n_in, n_out: 0.01*rng.randn(n_in, n_out)
        )

    fwd_fcn = mlp.compile()

    optimizer = SimpleGradientDescent(eta = 0.1)

    @symbolic_updater
    def train(x, y):
        output = mlp(x)
        cost = negative_log_likelihood_dangerous(output, y)
        updates = optimizer(cost, mlp.parameters)
        return updates

    train_fcn = train.compile()

    init_score = percent_argmax_correct(fwd_fcn(dataset.test_set.input), dataset.test_set.target)

    for x_m, y_m in zip_minibatch_iterate([dataset.training_set.input, dataset.training_set.target], minibatch_size=10, n_epochs=20):
        train_fcn(x_m, y_m)

    final_score = percent_argmax_correct(fwd_fcn(dataset.test_set.input), dataset.test_set.target)
    print 'Initial score: %s%%.  Final score: %s%%' % (init_score, final_score)
    assert init_score < 30
    assert final_score > 98


if __name__ == '__main__':

    test_mlp()