from general.numpy_helpers import get_rng
import numpy as np
from utils.predictors.i_predictor import IPredictor

__author__ = 'peter'


class ConditionalHerding(IPredictor):
    """
    The conditional herding predictor, as described in:

    On Herding and the Perceptron Cycling Theorem
    Andrew E. Gelfand, Yutian Chen, Max Welling, Laurens van der Maaten
    http://papers.nips.cc/paper/4004-on-herding-and-the-perceptron-cycling-theorem.pdf

    Note: This predictor will only work well if you average out test results over the course of
    training (see class RunningAverage).
    """

    def __init__(self, w, b, theta, alpha):
        """
        :param w: A (n_in, n_hidden) input-hidden weight matrix
        :param b: An (n_hidden, n_out) hidden-output weight matrix
        :param theta: A (n_hidden, ) hidden-bias vector
        :param alpha: A (n_out, ) output-bias vector
        :return:
        """

        n_hidden = w.shape[1]
        assert n_hidden == theta.shape[0] == b.shape[0]
        n_out = b.shape[1]
        assert n_out == b.shape[1] == alpha.shape[0]
        self.w = w
        self.b = b
        self.theta = theta
        self.alpha = alpha
        self.persistent_chain_initialized = False


    def predict(self, x):

        z_star = binarize(x.dot(self.w)+self.theta)
        y_star = binarize(z_star.dot(self.b)+self.alpha)
        # z_star, y_star = self.find_local_max(x)
        return y_star

    # def find_local_max(self, x):
    #
    #     z_star = None
    #     y_star = np.zeros((x.shape[0], len(self.alpha)))
        for i in xrange(10):
            z_star = binarize(x.dot(self.w) + self.theta + y_star.dot(self.b.T))
            y_star = binarize(z_star.dot(self.b)+self.alpha)
        return z_star, y_star

    def shake_it(self, x, y_init, n_steps=1):
        y_star = y_init
        for i in xrange(n_steps):
            z_star = binarize(x.dot(self.w) + self.theta + y_star.dot(self.b.T))
            y_star = binarize(z_star.dot(self.b)+self.alpha)
        return z_star, y_star

    def train(self, x, y):

        z = binarize(x.dot(self.w) + self.theta + y.dot(self.b.T))
        z_star = binarize(x.dot(self.w)+self.theta)
        y_star = binarize(z_star.dot(self.b)+self.alpha)

        # z_star, y_star = self.shake_it(x, y, 4)

        # print 'Potential of pos-states: %s, neg_states: %s' % \
        #       (self.compute_energy(x, z, y).mean(), self.compute_energy(x, z_star, y_star).mean())
        # Something's wrong - The negative states should always have higher potential than the positive, but we don't
        # see this.  I think we need to use a different procedure for computing z_star, y_star

        delta_w = x.T.dot(z) - x.T.dot(z_star)
        delta_theta = z.sum(axis=0) - z_star.sum(axis=0)
        delta_b = z.T.dot(y) - z_star.T.dot(y_star)
        delta_alpha = y.sum(axis=0) - y_star.sum(axis=0)

        self.w += delta_w / self.w.size
        self.theta += delta_theta / self.w.size
        self.b += delta_b / self.b.size
        self.alpha += delta_alpha / self.b.size

    def compute_energy(self, x, z, y):
        return np.einsum('si,ij,sj->s', x, self.w, z) + z.dot(self.theta) + np.einsum('si,ij,sj->s', z, self.b, y) + y.dot(self.alpha)

    @classmethod
    def from_initializer(cls, input_size, hidden_size, output_size, initial_mag = 1, rng = None):
        rng = get_rng(rng)
        return cls(
            w = initial_mag*rng.normal(size=(input_size, hidden_size)).astype(np.float32)/(input_size*hidden_size),
            theta = np.zeros(hidden_size, dtype = np.float32),
            b = initial_mag*rng.normal(size=(hidden_size, output_size)).astype(np.float32)/(hidden_size*output_size),
            alpha = np.zeros(output_size, dtype = np.float32),
            )


binarize = lambda x: np.sign(x)


if __name__ == '__main__':
    from utils.datasets.mnist import get_mnist_dataset
    from utils.benchmarks.predictor_comparison import assess_online_predictor
    from utils.tools.processors import RunningAverageWithBurnin
    # Here we try to replicate the results from Table 1, MNIST, CH, 100 hidden units.  They claim 2.09% error.
    # So far, nowhere close

    dataset = get_mnist_dataset(flat = True, binarize = -1).to_onehot(form = 'sign')

    predictor = ConditionalHerding.from_initializer(
        input_size = dataset.input_size,
        hidden_size=100,
        output_size=dataset.target_size
    )

    assess_online_predictor(
        predictor = predictor,
        dataset = dataset,
        minibatch_size=100,
        evaluation_function='percent_argmax_correct',
        test_epochs = np.linspace(0, 40, 100),
        accumulator=lambda: RunningAverageWithBurnin(burn_in_steps=20),
        )
