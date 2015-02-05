from collections import namedtuple
import numpy as np
from plato.interfaces.decorators import symbolic_updater, symbolic_stateless
import theano
from theano.tensor.shared_randomstreams import RandomStreams
from utils.benchmarks.compare_predictors import compare_predictors
from utils.benchmarks.plot_learning_curves import plot_learning_curves
from utils.datasets.datasets import DataSet, DataCollection
from utils.datasets.synthetic_logistic import get_logistic_regression_data
from utils.predictors.i_predictor import IPredictor
from utils.predictors.mock_predictor import MockPredictor
from utils.tools.mymath import bernoulli, sigm
import theano.tensor as tt

__author__ = 'peter'

"""
Here, we do logistic regression with binary weights.
"""


class SamplingPredictor(IPredictor):

    def __init__(self, sampler, mode = 'test_and_run'):
        self._train_function = sampler.update.compile(mode=mode)
        self._predict_function = sampler.sample_posterior.compile(mode=mode)

    def train(self, input_data, target_data):
        self._train_function(input_data, target_data)

    def predict(self, input_data):
        return self._predict_function(input_data)


SamplingRegressor = namedtuple('SamplingRegressor', ('update', 'sample_posterior'))


def simple_binary_gibbs_regressor(n_dim_in, n_dim_out, sample_y = False, seed = None):
    """
    Returns the simplest form of a binary regressor.

    :param n_dim_in: Number of dimensions of the input
    :param n_dim_out: Number of dimensions of the output
    :param sample_y: Sample output from a bernoulli distribution (T) or return the probability (F)
    :param seed: Seed for the random number generator.
    :return: A SamplingRegressor object containing the functions for updating and sampling the posterior.
    """

    w = theano.shared(np.zeros((n_dim_in, n_dim_out), dtype = 'int'), name = 'w')
    rng = RandomStreams(seed)
    alpha = theano.shared(np.array(0))  # scalar

    @symbolic_updater
    def update(x, y):
        w_0 = tt.set_subtensor(w[alpha], 0)  # (n_dim_in, n_dim_out)
        w_1 = tt.set_subtensor(w[alpha], 1)  # (n_dim_in, n_dim_out)
        z_0 = tt.nnet.sigmoid(x.dot(w_0))  # (n_samples, n_dim_out)
        z_1 = tt.nnet.sigmoid(x.dot(w_1))  # (n_samples, n_dim_out)
        log_likelihood_ratio = tt.sum(tt.log(bernoulli(y, z_1))-tt.log(bernoulli(y, z_0)), axis = 0)  # (n_dim_out, )
        p_wa = tt.nnet.sigmoid(log_likelihood_ratio)  # (n_dim_out, )
        print p_wa.tag.test_value
        w_sample = rng.binomial(p=p_wa)  # (n_dim_out, )
        w_new = tt.set_subtensor(w[alpha], w_sample)  # (n_dim_in, n_dim_out)
        # if 0.001<p_wa.tag.test_value<0.999:
        #     print 'Some uncertainty!'
        return [(w, w_new), (alpha, (alpha+1) % n_dim_in)]

    @symbolic_stateless
    def sample_posterior(x):
        p_y = tt.nnet.sigmoid(x.dot(w))
        return rng.binomial(p = p_y) if sample_y else p_y

    return SamplingRegressor(update=update, sample_posterior=sample_posterior)


def herded_binary_gibbs_regressor(n_dim_in, n_dim_out, sample_y = False, seed = None):

    w = theano.shared(np.zeros((n_dim_in, n_dim_out), dtype = 'int'), name = 'w')
    phi = theano.shared(np.zeros((n_dim_in, n_dim_out), dtype = 'float'), name = 'phi')

    rng = RandomStreams(seed)
    alpha = theano.shared(np.array(0))

    @symbolic_updater
    def update(x, y):
        w_0 = tt.set_subtensor(w[alpha], 0)  # (n_dim_in, n_dim_out)
        w_1 = tt.set_subtensor(w[alpha], 1)  # (n_dim_in, n_dim_out)
        z_0 = tt.nnet.sigmoid(x.dot(w_0))  # (n_samples, n_dim_out)
        z_1 = tt.nnet.sigmoid(x.dot(w_1))  # (n_samples, n_dim_out)
        log_likelihood_ratio = tt.sum(tt.log(bernoulli(y, z_1))-tt.log(bernoulli(y, z_0)), axis = 0)  # (n_dim_out, )
        p_wa = tt.nnet.sigmoid(log_likelihood_ratio)  # (n_dim_out, )

        # Now, the herding part... here're the 3 lines from the minipaper
        phi_alpha = phi[alpha] + p_wa
        w_sample = phi_alpha > 0.5
        new_phi_alpha = phi_alpha - w_sample

        new_phi = tt.set_subtensor(phi[alpha], new_phi_alpha)
        w_new = tt.set_subtensor(w[alpha], w_sample)  # (n_dim_in, n_dim_out)

        # showloc()
        return [(w, w_new), (phi, new_phi), (alpha, (alpha+1) % n_dim_in)]

    @symbolic_stateless
    def sample_posterior(x):
        p_y = tt.nnet.sigmoid(x.dot(w))
        return rng.binomial(p = p_y) if sample_y else p_y

    return SamplingRegressor(update=update, sample_posterior=sample_posterior)


def demo_binary_regression():

    n_steps = 500
    n_dims = 20
    n_training = 1000
    n_test = 100
    noise_factor = 0.1
    sample_y = False

    x_tr, y_tr, x_ts, y_ts, w_true = get_logistic_regression_data(n_dims = n_dims,
        n_training=n_training, n_test=n_test, noise_factor = noise_factor)
    dataset = DataSet(DataCollection(x_tr, y_tr), DataCollection(x_ts, y_ts))  #.process_with(targets_processor=lambda (x, ): (OneHotEncoding()(x[:, 0]), ))

    records = compare_predictors(
        dataset = dataset,
        offline_predictor_constuctors={
            'Optimal': lambda: MockPredictor(lambda x: sigm(x.dot(w_true))),
            },
        incremental_predictor_constructors = {
            'gibbs': lambda: SamplingPredictor(simple_binary_gibbs_regressor(
                n_dim_in=x_tr.shape[1],
                n_dim_out=y_tr.shape[1],
                sample_y = sample_y,
                seed = None,
                ), mode = 'debug'),
            # 'herded-gibbs': lambda: SamplingPredictor(herded_binary_gibbs_regressor(
            #     n_dim_in=x_tr.shape[1],
            #     n_dim_out=y_tr.shape[1],
            #     sample_y = sample_y,
            #     seed = None,
            #     ), mode = 'tr'),
            },
        test_points = np.arange(n_steps).astype('float'),
        evaluation_function = 'mse',
        report_test_scores=False
        )

    plot_learning_curves(records, title = 'Logistic Regression Dataset. \nn_training=%s, n_test=%s, n_dims=%s, noise_factor=%s, sample_y=%s'
        % (n_training, n_test, n_dims, noise_factor, sample_y))


if __name__ == '__main__':

    demo_binary_regression()
