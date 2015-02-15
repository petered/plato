import numpy as np
from plotting.live_plotting import LiveStream, LiveCanal
from utils.benchmarks.compare_predictors import compare_predictors
from utils.benchmarks.plot_learning_curves import plot_learning_curves
from utils.datasets.datasets import DataSet, DataCollection
from utils.datasets.synthetic_logistic import get_logistic_regression_data
from utils.predictors.i_predictor import IPredictor
from utils.predictors.mock_predictor import MockPredictor
from utils.tools.mymath import sigm
from plato.tools.sampling import GibbsRegressor, HerdedGibbsRegressor

__author__ = 'peter'

"""
Here, we do logistic regression with binary weights.
"""


class SamplingPredictor(IPredictor):

    def __init__(self, sampler, mode = 'test_and_run'):
        self.train_function = sampler.update.compile(mode=mode)
        self.predict_function = sampler.sample_posterior.compile(mode=mode)

    def train(self, input_data, target_data):
        self.train_function(input_data, target_data)

    def predict(self, input_data):
        return self.predict_function(input_data)


def demo_binary_regression():

    n_steps = 1000
    n_dims = 20
    n_training = 50
    n_test = 100
    noise_factor = 0.0
    sample_y = False
    plot = True

    x_tr, y_tr, x_ts, y_ts, w_true = get_logistic_regression_data(n_dims = n_dims,
        n_training=n_training, n_test=n_test, noise_factor = noise_factor)
    dataset = DataSet(DataCollection(x_tr, y_tr), DataCollection(x_ts, y_ts))  #.process_with(targets_processor=lambda (x, ): (OneHotEncoding()(x[:, 0]), ))

    def setup_visualization(predictor):
        """ Lets you plot internals of predictor in as it trains. """
        if isinstance(predictor, SamplingPredictor):
            # variable_getter = lambda: predictor.train_function.locals
            predictor.train_function.set_debug_variables('locals+class')

            def get_plotting_vals():
                lv = predictor.train_function.get_debug_values()
                return {
                    'alpha': lv['self._alpha'],
                    'w': lv['self._w'],
                    'p_wa': lv['p_wa'].squeeze(),
                    'y': lv['y'],
                    }
            predictor.train_function.add_callback(get_plotting_vals)

    records = compare_predictors(
        dataset = dataset,
        offline_predictor_constructors={
            'Optimal': lambda: MockPredictor(lambda x: sigm(x.dot(w_true))),
            },
        incremental_predictor_constructors = {
            'gibbs-single': lambda: SamplingPredictor(GibbsRegressor(
                n_dim_in=x_tr.shape[1],
                n_dim_out=y_tr.shape[1],
                sample_y = sample_y,
                n_alpha = 1,
                seed = None,
                ), mode = 'tr'),
            'batch-gibbs': lambda: SamplingPredictor(GibbsRegressor(
                n_dim_in=x_tr.shape[1],
                n_dim_out=y_tr.shape[1],
                sample_y = sample_y,
                n_alpha = 'all',
                seed = None,
                ), mode = 'tr'),
            'herded-gibbs': lambda: SamplingPredictor(HerdedGibbsRegressor(
                n_dim_in=x_tr.shape[1],
                n_dim_out=y_tr.shape[1],
                sample_y = sample_y,
                n_alpha = 1,
                seed = None,
                ), mode = 'tr'),
            'herded-batch-gibbs': lambda: SamplingPredictor(HerdedGibbsRegressor(
                n_dim_in=x_tr.shape[1],
                n_dim_out=y_tr.shape[1],
                sample_y = sample_y,
                n_alpha = 'all',
                seed = None,
                ), mode = 'tr'),
            },
        test_points = np.arange(n_steps).astype('float'),
        evaluation_function = 'mse',
        report_test_scores=False,
        on_construction_callback=setup_visualization if plot else None
        )

    plot_learning_curves(records, title = 'Logistic Regression Dataset. \nn_training=%s, n_test=%s, n_dims=%s, noise_factor=%s, sample_y=%s'
        % (n_training, n_test, n_dims, noise_factor, sample_y), xscale = 'symlog', yscale = 'linear')


if __name__ == '__main__':

    demo = 'compare'

    if demo == 'compare':
        demo_binary_regression()
    else:
        raise Exception()