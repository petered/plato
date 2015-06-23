from general.test_mode import is_test_mode, set_test_mode
from plato.interfaces.decorators import set_enable_omniscence
from plato.tools.dbn import DeepBeliefNet
from plato.tools.networks import StochasticLayer, FullyConnectedBridge
import numpy as np
from plato.tools.optimizers import SimpleGradientDescent
from plato.tools.tdb_plotting import tdbplot
from plotting.db_plotting import dbplot
from plotting.live_plotting import LiveStream
from utils.benchmarks.train_and_test import percent_argmax_correct
from utils.bureaucracy import minibatch_iterate
from utils.datasets.mnist import get_mnist_dataset
from utils.tools.processors import OneHotEncoding

__author__ = 'peter'


def demo_dbn_mnist(plot = True, test_mode = False):
    """
    In this demo we train an RBM on the MNIST input data (labels are ignored).  We plot the state of a markov chanin
    that is being simulaniously sampled from the RBM, and the parameters of the RBM.
    """

    set_enable_omniscence(True)
    minibatch_size = 20
    dataset = get_mnist_dataset().process_with(inputs_processor=lambda (x, ): (x.reshape(x.shape[0], -1), ))
    w_init = lambda n_in, n_out: 0.01 * np.random.randn(n_in, n_out)
    n_training_epochs_1 = 20
    n_training_epochs_2 = 20
    check_period = 300

    if test_mode:
        n_training_epochs_1 = 0.01
        n_training_epochs_2 = 0.01
        check_period=100

    dbn = DeepBeliefNet(
        layers = {
            'vis': StochasticLayer('bernoulli'),
            'hid': StochasticLayer('bernoulli'),
            'ass': StochasticLayer('bernoulli'),
            'lab': StochasticLayer('bernoulli'),
            },
        bridges = {
            ('vis', 'hid'): FullyConnectedBridge(w = w_init(784, 500), b_rev = 0),
            ('hid', 'ass'): FullyConnectedBridge(w = w_init(500, 500), b_rev = 0),
            ('lab', 'ass'): FullyConnectedBridge(w = w_init(10, 500), b_rev = 0)
        }
    )

    # Compile the functions you're gonna use.
    train_first_layer = dbn.get_constrastive_divergence_function(visible_layers = 'vis', hidden_layers='hid', optimizer=SimpleGradientDescent(eta = 0.01), n_gibbs = 1, persistent=True).compile()
    free_energy_of_first_layer = dbn.get_free_energy_function(visible_layers='vis', hidden_layers='hid').compile()
    train_second_layer = dbn.get_constrastive_divergence_function(visible_layers=('hid', 'lab'), hidden_layers='ass', input_layers=('vis', 'lab'), n_gibbs=1, persistent=True).compile()
    predict_label = dbn.get_inference_function(input_layers = 'vis', output_layers='lab', path = [('vis', 'hid'), ('hid', 'ass'), ('ass', 'lab')], smooth = True).compile()

    encode_label = OneHotEncoding(n_classes=10)

    # Step 1: Train the first layer, plotting the weights and persistent chain state.
    if plot:
        train_first_layer.set_debug_variables(lambda: {
                'weights': dbn._bridges['vis', 'hid']._w.T.reshape((-1, 28, 28)),
                'smooth_vis_state': dbn.get_inference_function('hid', 'vis', smooth = True).symbolic_stateless(*train_first_layer.locals()['initial_hidden']).reshape((-1, 28, 28))
            })
        plotter = LiveStream(train_first_layer.get_debug_values)

    for i, (n_samples, visible_data, label_data) in enumerate(dataset.training_set.minibatch_iterator(minibatch_size = minibatch_size, epochs = n_training_epochs_1, single_channel = True)):
        train_first_layer(visible_data)
        if i % check_period == 0:
            print 'Free Energy of Test Data: %s' % (free_energy_of_first_layer(dataset.test_set.input).mean())
            if plot:
                plotter.update()

    # Step 2: Train the second layer and simultanously compute the classification error from forward passes.
    if plot:
        train_second_layer.set_debug_variables(lambda: {
            'w_vis_hid': dbn._bridges['vis', 'hid']._w.T.reshape((-1, 28, 28)),
            'w_hid_ass': dbn._bridges['hid', 'ass']._w,
            'w_lab_ass': dbn._bridges['hid', 'ass']._w,
            'associative_state': train_second_layer.locals()['sleep_hidden'][0].reshape((-1, 20, 25)),
            'hidden_state': train_second_layer.locals()['sleep_visible'][0].reshape((-1, 20, 25)),
            'smooth_vis_state': dbn.get_inference_function('hid', 'vis', smooth = True).symbolic_stateless(train_second_layer.locals()['sleep_visible'][0]).reshape((-1, 28, 28))
            })
        plotter = LiveStream(train_first_layer.get_debug_values)

    for i, (n_samples, visible_data, label_data) in enumerate(dataset.training_set.minibatch_iterator(minibatch_size = minibatch_size, epochs = n_training_epochs_2, single_channel = True)):
        train_second_layer(visible_data, encode_label(label_data))
        if i % check_period == 0:
            out, = predict_label(dataset.test_set.input)
            score = percent_argmax_correct(actual = out, target = dataset.test_set.target)
            print 'Classification Score: %s' % score
            if plot:
                plotter.update()


def demo_dbn_unsupervised(
        minibatch_size = 20,
        n_training_epochs_1 = 20,
        n_training_epochs_2 = 20,
        plot = True,
        test_mode = False
        ):
    """
    In this demo we train an RBM on the MNIST input data (labels are ignored).  We plot the state of a markov chanin
    that is being simulaniously sampled from the RBM, and the parameters of the RBM.
    """

    set_enable_omniscence(True)

    dataset = get_mnist_dataset(flat = True)
    w_init = lambda n_in, n_out: 0.01 * np.random.randn(n_in, n_out)

    check_period = 10

    if is_test_mode():
        n_training_epochs_1 = 0.01
        n_training_epochs_2 = 0.01
        check_period=100

    dbn = DeepBeliefNet(
        layers = {
            'vis': StochasticLayer('bernoulli'),
            'hid1': StochasticLayer('bernoulli'),
            'hid2': StochasticLayer('gaussian'),
            },
        bridges = {
            ('vis', 'hid1'): FullyConnectedBridge(w = w_init(784, 500), b_rev = 0),
            ('hid1', 'hid2'): FullyConnectedBridge(w = w_init(500, 10), b_rev = 0),
        }
    )

    # Compile the functions you're gonna use.
    train_first_layer = dbn.get_constrastive_divergence_function(visible_layers = 'vis', hidden_layers='hid1', optimizer=SimpleGradientDescent(eta = 0.01), n_gibbs = 1, persistent=True).compile()
    train_second_layer = dbn.get_constrastive_divergence_function(visible_layers = 'hid1', hidden_layers='hid2', input_layers='vis', optimizer=SimpleGradientDescent(eta = 0.01), n_gibbs = 1, method = 'samples', persistent=True).compile()
    # free_energy_of_first_layer = dbn.get_free_energy_function(visible_layers='vis', hidden_layers='hid').compile()
    # train_second_layer = dbn.get_constrastive_divergence_function(visible_layers=('hid', 'lab'), hidden_layers='ass', input_layers=('vis', 'lab'), n_gibbs=1, persistent=True).compile()
    # predict_label = dbn.get_inference_function(input_layers = 'vis', output_layers='lab', path = [('vis', 'hid'), ('hid', 'ass'), ('ass', 'lab')], smooth = True).compile()

    print 'fdsfds'
    for i, vis_data in enumerate(minibatch_iterate(dataset.training_set.input, minibatch_size=minibatch_size, n_epochs=n_training_epochs_1)):
        if i % check_period == 0:
            dbplot(dbn._bridges['vis', 'hid1']._w.get_value().T.reshape((-1, 28, 28)), 'w')
        print i
        train_first_layer(vis_data)

    for i, vis_data in enumerate(minibatch_iterate(dataset.training_set.input, minibatch_size=minibatch_size, n_epochs=n_training_epochs_2)):
        train_second_layer(vis_data)

    #
    # # Step 1: Train the first layer, plotting the weights and persistent chain state.
    # if plot:
    #     train_first_layer.set_debug_variables(lambda: {
    #             'weights': dbn._bridges['vis', 'hid']._w.T.reshape((-1, 28, 28)),
    #             'smooth_vis_state': dbn.get_inference_function('hid', 'vis', smooth = True).symbolic_stateless(*train_first_layer.locals()['initial_hidden']).reshape((-1, 28, 28))
    #         })
    #     plotter = LiveStream(train_first_layer.get_debug_values)
    #
    # for i, (n_samples, visible_data, label_data) in enumerate(dataset.training_set.minibatch_iterator(minibatch_size = minibatch_size, epochs = n_training_epochs_1, single_channel = True)):
    #     train_first_layer(visible_data)
    #     if i % check_period == 0:
    #         print 'Free Energy of Test Data: %s' % (free_energy_of_first_layer(dataset.test_set.input).mean())
    #         if plot:
    #             plotter.update()
    #
    # # Step 2: Train the second layer and simultanously compute the classification error from forward passes.
    # if plot:
    #     train_second_layer.set_debug_variables(lambda: {
    #         'w_vis_hid': dbn._bridges['vis', 'hid']._w.T.reshape((-1, 28, 28)),
    #         'w_hid_ass': dbn._bridges['hid', 'ass']._w,
    #         'w_lab_ass': dbn._bridges['hid', 'ass']._w,
    #         'associative_state': train_second_layer.locals()['sleep_hidden'][0].reshape((-1, 20, 25)),
    #         'hidden_state': train_second_layer.locals()['sleep_visible'][0].reshape((-1, 20, 25)),
    #         'smooth_vis_state': dbn.get_inference_function('hid', 'vis', smooth = True).symbolic_stateless(train_second_layer.locals()['sleep_visible'][0]).reshape((-1, 28, 28))
    #         })
    #     plotter = LiveStream(train_first_layer.get_debug_values)
    #
    # for i, (n_samples, visible_data, label_data) in enumerate(dataset.training_set.minibatch_iterator(minibatch_size = minibatch_size, epochs = n_training_epochs_2, single_channel = True)):
    #     train_second_layer(visible_data, encode_label(label_data))
    #     if i % check_period == 0:
    #         out, = predict_label(dataset.test_set.input)
    #         score = percent_argmax_correct(actual = out, target = dataset.test_set.target)
    #         print 'Classification Score: %s' % score
    #         if plot:
    #             plotter.update()


EXPERIMENTS = dict()

EXPERIMENTS['associative'] = demo_dbn_mnist

EXPERIMENTS['unsupervised'] = demo_dbn_unsupervised


if __name__ == '__main__':

    set_test_mode(True)
    EXPERIMENTS['unsupervised']()

