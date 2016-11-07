import numpy as np

from artemis.general.test_mode import is_test_mode
from plato.core import EnableOmniscence
from plato.tools.dbn.dbn import DeepBeliefNet
from plato.tools.rbm.rbm_parts import StochasticNonlinearity, FullyConnectedBridge
from plato.tools.optimization.optimizers import SimpleGradientDescent
from artemis.plotting.db_plotting import dbplot
from artemis.ml.predictors.train_and_test import percent_argmax_correct
from artemis.ml.datasets.mnist import get_mnist_dataset
from artemis.ml.tools.processors import OneHotEncoding


__author__ = 'peter'


def demo_dbn_mnist(plot = True):
    """
    In this demo we train an RBM on the MNIST input data (labels are ignored).  We plot the state of a markov chanin
    that is being simulaniously sampled from the RBM, and the parameters of the RBM.
    """

    minibatch_size = 20
    dataset = get_mnist_dataset().process_with(inputs_processor=lambda (x, ): (x.reshape(x.shape[0], -1), ))
    w_init = lambda n_in, n_out: 0.01 * np.random.randn(n_in, n_out)
    n_training_epochs_1 = 20
    n_training_epochs_2 = 20
    check_period = 300

    with EnableOmniscence():  # This constrction allows us to access internal variables for plotting purposes.  When you
        # call fcn.locals() on a symbolic function you can get the values of their variables.

        if is_test_mode():
            n_training_epochs_1 = 0.01
            n_training_epochs_2 = 0.01
            check_period=100

        dbn = DeepBeliefNet(
            layers = {
                'vis': StochasticNonlinearity('bernoulli'),
                'hid': StochasticNonlinearity('bernoulli'),
                'ass': StochasticNonlinearity('bernoulli'),
                'lab': StochasticNonlinearity('bernoulli'),
                },
            bridges = {
                ('vis', 'hid'): FullyConnectedBridge(w = w_init(784, 500), b_rev = 0),
                ('hid', 'ass'): FullyConnectedBridge(w = w_init(500, 500), b_rev = 0),
                ('lab', 'ass'): FullyConnectedBridge(w = w_init(10, 500), b_rev = 0)
            }
        )

        # Compile the functions you're gonna use.
        train_first_layer = dbn.get_constrastive_divergence_function(visible_layers = 'vis', hidden_layers='hid', optimizer=SimpleGradientDescent(eta = 0.01), n_gibbs = 1, persistent=True).compile(add_test_values = True)
        free_energy_of_first_layer = dbn.get_free_energy_function(visible_layers='vis', hidden_layers='hid').compile(add_test_values = True)
        train_second_layer = dbn.get_constrastive_divergence_function(visible_layers=('hid', 'lab'), hidden_layers='ass', input_layers=('vis', 'lab'), n_gibbs=1, persistent=True).compile(add_test_values = True)
        predict_label = dbn.get_inference_function(input_layers = 'vis', output_layers='lab', path = [('vis', 'hid'), ('hid', 'ass'), ('ass', 'lab')], smooth = True).compile(add_test_values = True)

        encode_label = OneHotEncoding(n_classes=10)

        # Step 1: Train the first layer, plotting the weights and persistent chain state.
        for i, (n_samples, visible_data, label_data) in enumerate(dataset.training_set.minibatch_iterator(minibatch_size = minibatch_size, epochs = n_training_epochs_1, single_channel = True)):
            train_first_layer(visible_data)
            if i % check_period == 0:
                print 'Free Energy of Test Data: %s' % (free_energy_of_first_layer(dataset.test_set.input).mean())
                if plot:
                    dbplot(dbn.bridges['vis', 'hid'].w.get_value().T.reshape((-1, 28, 28)), 'weights')
                    dbplot(train_first_layer.locals()['sleep_visible'][0].reshape((-1, 28, 28)), 'vis_sleep_state')

        # Step 2: Train the second layer and simultanously compute the classification error from forward passes.
        for i, (n_samples, visible_data, label_data) in enumerate(dataset.training_set.minibatch_iterator(minibatch_size = minibatch_size, epochs = n_training_epochs_2, single_channel = True)):
            train_second_layer(visible_data, encode_label(label_data))
            if i % check_period == 0:
                out, = predict_label(dataset.test_set.input)
                score = percent_argmax_correct(actual = out, target = dataset.test_set.target)
                print 'Classification Score: %s' % score
                if plot:
                    dbplot(dbn.bridges['vis', 'hid'].w.get_value().T.reshape((-1, 28, 28)), 'w_vis_hid')
                    dbplot(dbn.bridges['hid', 'ass'].w.get_value(), 'w_hid_ass')
                    dbplot(dbn.bridges['hid', 'ass'].w.get_value(), 'w_lab_ass')
                    dbplot(train_second_layer.locals()['sleep_visible'][0].reshape((-1, 20, 25)), 'hidden_state')


if __name__ == '__main__':

    demo_dbn_mnist()
