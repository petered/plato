from general.test_mode import is_test_mode, set_test_mode
from plato.tools.optimizers import SimpleGradientDescent
from plotting.db_plotting import dbplot
from utils.bureaucracy import minibatch_iterate
from utils.datasets.mnist import get_mnist_dataset
import numpy as np
from utils.tools.stacked_dbn import StackedDeepBeliefNet, BernoulliBernoulliRBM, BernoulliGaussianRBM


def demo_simple_dbn(
        minibatch_size = 10,
        n_training_epochs_1 = 5,
        n_training_epochs_2 = 5,\
        n_hidden_1 = 500,
        n_hidden_2 = 10,
        seed = None
):

    # set_enable_omniscence(True)

    dataset = get_mnist_dataset(flat = True)
    rng = np.random.RandomState(seed)
    w_init = lambda shape: 0.01 * rng.randn(*shape)


    check_period = 100

    if is_test_mode():
        n_training_epochs_1 = 0.01
        n_training_epochs_2 = 0.01
        check_period=100

    dbn1 = StackedDeepBeliefNet(
        rbms = [BernoulliBernoulliRBM.from_initializer(n_visible = 784, n_hidden=500, w_init_fcn = w_init)]
    )

    train_first_layer = dbn1.get_training_fcn(optimizer=SimpleGradientDescent(eta = 0.01), n_gibbs = 1, persistent=True).compile()
    sample_first_layer = dbn1.get_sampling_fcn(initial_vis=dataset.training_set.input[:minibatch_size], n_steps = 10).compile()
    for i, vis_data in enumerate(minibatch_iterate(dataset.training_set.input, minibatch_size=minibatch_size, n_epochs=n_training_epochs_1)):
        if i % check_period == 0:
            dbplot(dbn1.rbms[0].w.get_value().T[:100].reshape([-1, 28, 28]), 'weights1')
            dbplot(sample_first_layer()[0].reshape(-1, 28, 28), 'samples1')
        train_first_layer(vis_data)

    dbn2 = dbn1.stack_another(rbm = BernoulliGaussianRBM.from_initializer(n_visible=n_hidden_1, n_hidden=n_hidden_2, w_init_fcn=w_init))
    train_second_layer = dbn2.get_training_fcn(optimizer=SimpleGradientDescent(eta = 0.001), n_gibbs = 1, persistent=True).compile()
    sample_second_layer = dbn2.get_sampling_fcn(initial_vis=dataset.training_set.input[:minibatch_size], n_steps = 10).compile()
    for i, vis_data in enumerate(minibatch_iterate(dataset.training_set.input, minibatch_size=minibatch_size, n_epochs=n_training_epochs_1)):
        if i % check_period == 0:
            dbplot(dbn2.rbms[1].w.get_value(), 'weights2')
            dbplot(sample_second_layer()[0].reshape(-1, 28, 28), 'samples2')
        train_second_layer(vis_data)

    project_to_latent = dbn2.propup.compile(fixed_args = dict(stochastic = False))
    latent_test_data = project_to_latent(dataset.test_set.input)
    print 'Projected the test data to a latent space.  Shape: %s' % (latent_test_data.shape, )


if __name__ == '__main__':

    set_test_mode(False)
    demo_simple_dbn()