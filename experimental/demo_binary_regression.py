from collections import namedtuple
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


def setup_visualization(predictor):
    """ Lets you plot internals of predictor in as it trains. """
    if isinstance(predictor, SamplingPredictor):
        # variable_getter = lambda: predictor.train_function.locals
        predictor.train_function.set_debug_variables('locals+class')

        def get_plotting_vals():
            lv = predictor.train_function.get_debug_values()
            plot_dict = {
                'alpha': lv['self._alpha'],
                'w': lv['self._w'],
                'p_wa': lv['p_wa'].squeeze(),
                'y': lv['y'],
                }
            if 'self._phi' in lv:
                plot_dict['phi'] = lv['self._phi'].squeeze()

            print plot_dict['w'].T
            return plot_dict
        plotter = LiveStream(get_plotting_vals)

        predictor.train_function.add_callback(plotter.update)


DataParams = namedtuple('DataParams', ['x_tr', 'y_tr', 'x_ts', 'y_ts', 'w_true', 'n_dims', 'n_training', 'n_test', 'noise_factor'])


def get_data_for_figure(which_figure):

    if which_figure[0] in ('1', '2'):
        n_dims = 20
        n_training = 1000
        n_test = 100
        noise_factor = {
            '1': 0.1,
            '2A': 0.1,
            '2B': 0.1,
            '2C': 0.0,
            '2D': 1.0
            }[which_figure]
    elif which_figure[0] in ('3', '4'):
        n_dims = 20
        n_training = 50
        n_test = 100
        noise_factor = 0.0
    elif which_figure[0] == '5':
        n_dims = 20
        n_training = {'5A': 50, '5B': 50, '5C': 1000, '5D': 1000}[which_figure]
        n_test = 100
        noise_factor = 0.0
    elif which_figure[0] == 'X':
        n_dims = 20
        n_training = 500
        n_test = 100
        noise_factor = 0.0
    else:
        raise Exception('No configuration for figure "%s"' % (which_figure, ))

    x_tr, y_tr, x_ts, y_ts, w_true = get_logistic_regression_data(n_dims = n_dims,
    n_training=n_training, n_test=n_test, noise_factor = noise_factor)

    return DataParams(x_tr = x_tr, y_tr = y_tr, x_ts = x_ts, y_ts = y_ts, w_true = w_true, n_dims = n_dims, n_training = n_training,
        n_test = n_test, noise_factor = noise_factor)


def demo_binary_regression(which_figure, test_mode = False, plot = False):
    """
    Code for creating plots in our report.

    :param which_figure: Which figure of the report to replicate.  Or "X" for just
       experimenting with stuff.  The report is here:
       https://docs.google.com/document/d/1zBQdI-1tcEvqmizCuL2GX_ceqhLnOvGoN8ljciw9uGE/edit?usp=sharing
    :param test_mode: Just makes things run really fast to assert that they don't break.
    """

    n_steps = \
        3 if test_mode else \
        1000 if which_figure[0] in ('1', '2', '3', '4') else \
        10000 if which_figure[0] == '5' else \
        100000

    sample_y = False

    d = get_data_for_figure(which_figure)
    dataset = DataSet(DataCollection(d.x_tr, d.y_tr), DataCollection(d.x_ts, d.y_ts))

    def get_regressor_constructor(sampling_type, n_alpha, alpha_update_policy = 'sequential', possible_ws = (0, 1)):
        klass = {'gibbs': GibbsRegressor, 'herded': HerdedGibbsRegressor}[sampling_type]
        return lambda: SamplingPredictor(klass(
                n_dim_in=d.x_tr.shape[1],
                n_dim_out=d.y_tr.shape[1],
                sample_y = sample_y,
                n_alpha = n_alpha,
                seed = None,
                alpha_update_policy = alpha_update_policy,
                possible_ws = possible_ws
                ), mode = 'tr')

    full_set_of_regressors = {
        'gibbs': get_regressor_constructor('gibbs', n_alpha = 1, alpha_update_policy='sequential'),
        'herded': get_regressor_constructor('herded', n_alpha = 1, alpha_update_policy='sequential'),
        'gibbs-rand': get_regressor_constructor('gibbs', n_alpha = 1, alpha_update_policy='random'),
        'herded-rand': get_regressor_constructor('herded', n_alpha = 1, alpha_update_policy='random'),
        'gibbs-1/4': get_regressor_constructor('gibbs', n_alpha = 5, alpha_update_policy='sequential'),
        'herded-1/4': get_regressor_constructor('herded', n_alpha = 5, alpha_update_policy='sequential'),
        'gibbs-1/4-rand': get_regressor_constructor('gibbs', n_alpha = 5, alpha_update_policy='random'),
        'herded-1/4-rand': get_regressor_constructor('herded', n_alpha = 5, alpha_update_policy='random'),
        'gibbs-1/2': get_regressor_constructor('gibbs', n_alpha = 10, alpha_update_policy='sequential'),
        'herded-1/2': get_regressor_constructor('herded', n_alpha = 10, alpha_update_policy='sequential'),
        'gibbs-1/2-rand': get_regressor_constructor('gibbs', n_alpha = 10, alpha_update_policy='random'),
        'herded-1/2-rand': get_regressor_constructor('herded', n_alpha = 10, alpha_update_policy='random'),
        'gibbs-full': get_regressor_constructor('gibbs', n_alpha = 20, alpha_update_policy='sequential'),
        'herded-full': get_regressor_constructor('herded', n_alpha = 20, alpha_update_policy='sequential'),
        'gibbs-full-rand': get_regressor_constructor('gibbs', n_alpha = 20, alpha_update_policy='random'),
        'herded-full-rand': get_regressor_constructor('herded', n_alpha = 20, alpha_update_policy='random'),
        'gibbs-5choice': get_regressor_constructor('gibbs', n_alpha = 1, alpha_update_policy='sequential', possible_ws=[0, 0.25, 0.5, 0.75, 1]),
        'herded-5choice': get_regressor_constructor('herded', n_alpha = 1, alpha_update_policy='sequential', possible_ws=[0, 0.25, 0.5, 0.75, 1]),
        'gibbs-20choice': get_regressor_constructor('gibbs', n_alpha = 1, alpha_update_policy='sequential', possible_ws=np.linspace(0, 1, 20)),
        'herded-20choice': get_regressor_constructor('herded', n_alpha = 1, alpha_update_policy='sequential', possible_ws=np.linspace(0, 1, 20)),
        }

    regressors_to_compare = {
        'X': ['gibbs', 'gibbs-5choice', 'gibbs-20choice'],
        '1': ['gibbs'],
        '2A': ['gibbs', 'herded'],
        '2B': ['gibbs', 'herded'],
        '2C': ['gibbs', 'herded'],
        '2D': ['gibbs', 'herded'],
        '3A': ['gibbs', 'gibbs-1/4', 'gibbs-1/2', 'gibbs-full'],
        '3B': ['herded', 'herded-1/4', 'herded-1/2', 'herded-full'],
        '4A': ['gibbs', 'gibbs-rand', 'herded', 'herded-rand'],
        '4B': ['gibbs-1/4', 'gibbs-1/4-rand', 'herded-1/4', 'herded-1/4-rand'],
        '4C': ['gibbs-1/2', 'gibbs-1/2-rand', 'herded-1/2', 'herded-1/2-rand'],
        '4D': ['gibbs-full', 'gibbs-full-rand', 'herded-full', 'herded-full-rand'],
        '5A': ['gibbs', 'gibbs-5choice', 'gibbs-20choice'],
        '5B': ['herded', 'herded-5choice', 'herded-20choice'],
        '5C': ['gibbs', 'gibbs-5choice', 'gibbs-20choice'],
        '5D': ['herded', 'herded-5choice', 'herded-20choice'],
        }

    records = compare_predictors(
        dataset = dataset,
        offline_predictor_constructors={
            'Optimal': lambda: MockPredictor(lambda x: sigm(x.dot(d.w_true))),
            },
        incremental_predictor_constructors = {k: full_set_of_regressors[k] for k in regressors_to_compare[which_figure]},
        test_points = np.arange(n_steps).astype('float'),
        evaluation_function = 'mse',
        report_test_scores=False,
        on_construction_callback=setup_visualization if plot else None
        )

    plot_learning_curves(records, title = 'Logistic Regression Dataset. \nn_training=%s, n_test=%s, n_dims=%s, noise_factor=%s, sample_y=%s'
        % (d.n_training, d.n_test, d.n_dims, d.noise_factor, sample_y), xscale = 'symlog', yscale = 'linear', hang = not test_mode)


if __name__ == '__main__':

    figure = '5D'
    plot = False

    demo_binary_regression(figure, plot=plot)
