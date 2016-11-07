import numpy as np

from artemis.general.test_mode import is_test_mode, set_test_mode
from plato.tools.optimization.optimizers import SimpleGradientDescent
from artemis.plotting.db_plotting import dbplot
from artemis.ml.tools.iteration import minibatch_iterate
from artemis.ml.datasets.mnist import get_mnist_dataset
from plato.tools.dbn.stacked_dbn import StackedDeepBeliefNet, BernoulliBernoulliRBM, BernoulliGaussianRBM


def demo_simple_dbn(
        minibatch_size = 10,
        n_training_epochs_1 = 5,
        n_training_epochs_2 = 50,
        n_hidden_1 = 500,
        n_hidden_2 = 10,
        plot_period = 100,
        eta1 = 0.01,
        eta2 = 0.0001,
        w_init_mag_1 = 0.01,
        w_init_mag_2 = 0.5,
        seed = None
        ):
    """
    Train a DBN, and create a function to project the test data into a latent space

    :param minibatch_size:
    :param n_training_epochs_1: Number of training epochs for the first-level RBM
    :param n_training_epochs_2: Number of training epochs for the second-level RBM
    :param n_hidden_1: Number of hidden units for first RBM
    :param n_hidden_2:nNumber of hidden units for second RBM
    :param plot_period: How often to plot
    :param seed:
    :return:
    """

    dataset = get_mnist_dataset(flat = True)
    rng = np.random.RandomState(seed)
    w_init_1 = lambda shape: w_init_mag_1 * rng.randn(*shape)
    w_init_2 = lambda shape: w_init_mag_2 * rng.randn(*shape)

    if is_test_mode():
        n_training_epochs_1 = 0.01
        n_training_epochs_2 = 0.01

    # Train the first RBM
    dbn1 = StackedDeepBeliefNet(rbms = [BernoulliBernoulliRBM.from_initializer(n_visible = 784, n_hidden=n_hidden_1, w_init_fcn = w_init_1)])
    train_first_layer = dbn1.get_training_fcn(optimizer=SimpleGradientDescent(eta = eta1), n_gibbs = 1, persistent=True).compile()
    sample_first_layer = dbn1.get_sampling_fcn(initial_vis=dataset.training_set.input[:minibatch_size], n_steps = 10).compile()
    for i, vis_data in enumerate(minibatch_iterate(dataset.training_set.input, minibatch_size=minibatch_size, n_epochs=n_training_epochs_1)):
        if i % plot_period == plot_period-1:
            dbplot(dbn1.rbms[0].w.get_value().T[:100].reshape([-1, 28, 28]), 'weights1')
            dbplot(sample_first_layer()[0].reshape(-1, 28, 28), 'samples1')
        train_first_layer(vis_data)

    # Train the second RBM
    dbn2 = dbn1.stack_another(rbm = BernoulliGaussianRBM.from_initializer(n_visible=n_hidden_1, n_hidden=n_hidden_2, w_init_fcn=w_init_2))
    train_second_layer = dbn2.get_training_fcn(optimizer=SimpleGradientDescent(eta = eta2), n_gibbs = 1, persistent=True).compile()
    sample_second_layer = dbn2.get_sampling_fcn(initial_vis=dataset.training_set.input[:minibatch_size], n_steps = 10).compile()
    for i, vis_data in enumerate(minibatch_iterate(dataset.training_set.input, minibatch_size=minibatch_size, n_epochs=n_training_epochs_2)):
        if i % plot_period == 0:
            dbplot(dbn2.rbms[1].w.get_value(), 'weights2')
            dbplot(sample_second_layer()[0].reshape(-1, 28, 28), 'samples2')
        train_second_layer(vis_data)

    # Project data to latent space.
    project_to_latent = dbn2.propup.compile(fixed_args = dict(stochastic = False))
    latent_test_data = project_to_latent(dataset.test_set.input)
    print 'Projected the test data to a latent space.  Shape: %s' % (latent_test_data.shape, )

    decode = dbn2.propdown.compile(fixed_args = dict(stochastic = False))
    recon_test_data = decode(latent_test_data)
    print 'Reconstructed the test data.  Shape: %s' % (recon_test_data.shape, )


if __name__ == '__main__':

    set_test_mode(False)
    demo_simple_dbn()
