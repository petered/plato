from artemis.experiments.experiment_record import experiment_root, capture_created_experiments, ExperimentFunction
from artemis.experiments.ui import browse_experiments
from artemis.general.numpy_helpers import get_rng
from artemis.ml.datasets.mnist import get_mnist_dataset
from artemis.ml.predictors.train_and_test import train_and_test_online_predictor
from plato.tools.common.online_predictors import GradientBasedPredictor
from plato.tools.fa.direct_feedback_alignment import create_direct_feedback_alignment_net
from plato.tools.fa.feedback_alignment import create_feedback_alignment_net
from plato.tools.mlp.mlp import MultiLayerPerceptron
from plato.tools.optimization.optimizers import GradientDescent


def create_network(version, layer_sizes, optimizer, nonlinearity, final_nonlinearity, backwards_nonlinearity, loss,
                   w_init = 'xavier', rng=None):

    if version == 'fa':
        return create_feedback_alignment_net(layer_sizes=layer_sizes, optimizer=optimizer, backwards_nonlinearity=backwards_nonlinearity,
            nonlinearity=nonlinearity, final_nonlinearity=final_nonlinearity, loss=loss, w_init=w_init, rng=rng)
    elif version == 'dfa':
        return create_direct_feedback_alignment_net(layer_sizes=layer_sizes, optimizer=optimizer, backwards_nonlinearity=backwards_nonlinearity,
            nonlinearity=nonlinearity, final_nonlinearity=final_nonlinearity, loss=loss, w_init=w_init,  rng=rng)
    elif version == 'mlp':
        return GradientBasedPredictor(
            function=MultiLayerPerceptron.from_init(
                layer_sizes=layer_sizes,
                hidden_activations = nonlinearity,
                output_activation = final_nonlinearity,
                w_init=w_init,
                rng=rng,
                ),

            optimizer=optimizer,
            cost_function=loss,
            )
    else:
        raise Exception('No network version "{}"'.format(version))


@ExperimentFunction(is_root=True, one_liner_results=lambda scores: scores.get_oneliner())
def demo_feedback_alignment_mnist(
            version = 'fa',
            hidden_sizes = [100],
            nonlinearity = 'relu',
            final_nonlinearity = 'linear',
            loss = 'logistic-xe',
            backwards_nonlinearity = 'deriv',
            n_epochs=10,
            minibatch_size = 10,
            learning_rate = 0.01,
            seed = 1234,
            ):

    assert version in ('fa', 'dfa', 'mlp')
    rng = get_rng(seed)
    mnist = get_mnist_dataset(flat=True).to_onehot()

    nnet = create_network(
        version=version,
        layer_sizes=[mnist.input_size]+hidden_sizes+[10],
        optimizer=GradientDescent(learning_rate),
        backwards_nonlinearity=backwards_nonlinearity, nonlinearity=nonlinearity, final_nonlinearity=final_nonlinearity, loss=loss, rng=rng
        )

    training_info = train_and_test_online_predictor(
        dataset = mnist,
        train_fcn=nnet.train.compile(add_test_values = True),
        predict_fcn=nnet.predict.compile(add_test_values = True),
        minibatch_size=minibatch_size,
        n_epochs=n_epochs,
        test_epochs=('every', 0.5),
        )

    return training_info


with capture_created_experiments() as exs:
    demo_feedback_alignment_mnist.add_variant(version='fa')
    demo_feedback_alignment_mnist.add_variant(version='dfa')
    demo_feedback_alignment_mnist.add_variant(version='mlp')

for e in exs:
    e.add_variant(hidden_sizes=[200, 200, 200], n_epochs = 50)
for e in exs:
    e.add_variant(hidden_sizes=[500, 500, 500, 500, 500], n_epochs = 50)


if __name__ == '__main__':
    browse_experiments()
    # demo_feedback_alignment_mnist(version = 'dfa', hidden_sizes=[200, 200, 200])

