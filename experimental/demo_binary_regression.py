from collections import OrderedDict, namedtuple
from experimental.rf_ensembles import MockModePredictor, get_mnist_rf_ensemble_dataset
import numpy as np
from plotting.live_plotting import LiveStream
from utils.benchmarks.compare_predictors import compare_predictors
from utils.benchmarks.plot_learning_curves import plot_learning_curves
from utils.datasets.synthetic_logistic import get_logistic_regression_dataset
from utils.predictors.mock_predictor import MockPredictor
from utils.tools.mymath import sigm
from plato.tools.sampling import GibbsRegressor, HerdedGibbsRegressor, SamplingPredictor


Figure = namedtuple('Figure', ['id', 'number', 'letter', 'alias'])


__author__ = 'peter'

"""
Here, we do logistic regression with binary weights.

This is an exersize in organization.  There are 3 levels at which we can define our plot:

1) Just say the figure numner
2) Give high-level "user" commands
3) The actual predictors and objects to plot.

1 should translate to 2 should translate to 3.
"""


def setup_visualization(predictor):
    """ Lets you plot internals of predictor as it trains. """
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

        plotter = LiveStream(get_plotting_vals)
        predictor.train_function.add_callback(plotter.update)


def bad_value(value):
    raise ValueError('Bad Value: %s' % value)


def get_predictor_factory_factory(n_dim_in, n_dim_out, sample_y=False):
    
    def get_predictor_factory(sampling_type, n_alpha, alpha_update_policy = 'sequential', possible_ws = (0, 1)):
        klass = {'gibbs': GibbsRegressor, 'herded': HerdedGibbsRegressor}[sampling_type]
        return lambda: SamplingPredictor(klass(
                n_dim_in=n_dim_in,
                n_dim_out=n_dim_out,
                sample_y = sample_y,
                n_alpha = n_alpha,
                seed = None,
                alpha_update_policy = alpha_update_policy,
                possible_ws = possible_ws
                ), mode = 'tr')

    return get_predictor_factory


def get_named_predictors(names, n_dim_in, n_dim_out, sample_y = False, w_range = (0, 1)):
    
    predictor_factory_factory = get_predictor_factory_factory(n_dim_in, n_dim_out, sample_y)

    w_min, w_max = w_range

    full_set_of_predictors = {
        'gibbs': predictor_factory_factory('gibbs', n_alpha = 1, alpha_update_policy='sequential', possible_ws = (w_min, w_max)),
        'herded': predictor_factory_factory('herded', n_alpha = 1, alpha_update_policy='sequential', possible_ws = (w_min, w_max)),
        'gibbs-rand': predictor_factory_factory('gibbs', n_alpha = 1, alpha_update_policy='random', possible_ws = (w_min, w_max)),
        'herded-rand': predictor_factory_factory('herded', n_alpha = 1, alpha_update_policy='random', possible_ws = (w_min, w_max)),
        'gibbs-1/4': predictor_factory_factory('gibbs', n_alpha = n_dim_in/4, alpha_update_policy='sequential', possible_ws = (w_min, w_max)),
        'herded-1/4': predictor_factory_factory('herded', n_alpha = n_dim_in/4, alpha_update_policy='sequential', possible_ws = (w_min, w_max)),
        'gibbs-1/4-rand': predictor_factory_factory('gibbs', n_alpha = n_dim_in/4, alpha_update_policy='random', possible_ws = (w_min, w_max)),
        'herded-1/4-rand': predictor_factory_factory('herded', n_alpha = n_dim_in/4, alpha_update_policy='random', possible_ws = (w_min, w_max)),
        'gibbs-1/2': predictor_factory_factory('gibbs', n_alpha = n_dim_in/2, alpha_update_policy='sequential', possible_ws = (w_min, w_max)),
        'herded-1/2': predictor_factory_factory('herded', n_alpha = n_dim_in/2, alpha_update_policy='sequential', possible_ws = (w_min, w_max)),
        'gibbs-1/2-rand': predictor_factory_factory('gibbs', n_alpha = n_dim_in/2, alpha_update_policy='random', possible_ws = (w_min, w_max)),
        'herded-1/2-rand': predictor_factory_factory('herded', n_alpha = n_dim_in/2, alpha_update_policy='random', possible_ws = (w_min, w_max)),
        'gibbs-full': predictor_factory_factory('gibbs', n_alpha = n_dim_in, alpha_update_policy='sequential', possible_ws = (w_min, w_max)),
        'herded-full': predictor_factory_factory('herded', n_alpha = n_dim_in, alpha_update_policy='sequential', possible_ws = (w_min, w_max)),
        'gibbs-full-rand': predictor_factory_factory('gibbs', n_alpha = n_dim_in, alpha_update_policy='random', possible_ws = (w_min, w_max)),
        'herded-full-rand': predictor_factory_factory('herded', n_alpha = n_dim_in, alpha_update_policy='random', possible_ws = (w_min, w_max)),
        'gibbs-5choice': predictor_factory_factory('gibbs', n_alpha = 1, alpha_update_policy='sequential', possible_ws=np.linspace(w_min, w_max, 5)),
        'herded-5choice': predictor_factory_factory('herded', n_alpha = 1, alpha_update_policy='sequential', possible_ws=np.linspace(w_min, w_max, 5)),
        'gibbs-20choice': predictor_factory_factory('gibbs', n_alpha = 1, alpha_update_policy='sequential', possible_ws=np.linspace(w_min, w_max, 20)),
        'herded-20choice': predictor_factory_factory('herded', n_alpha = 1, alpha_update_policy='sequential', possible_ws=np.linspace(w_min, w_max, 20)),
        }
    
    predictors_to_compare = OrderedDict((k, full_set_of_predictors[k]) for k in names)

    return predictors_to_compare


def lookup_figure(figure_id_or_alias):
    aliases = {
        'mnist-ensemble': '6A',
        'mnist-ensemble-full': '6B',
        }
    reverse_aliases = {fig_id: alias for alias, fig_id in aliases.iteritems()}
    assert set(reverse_aliases.viewkeys()).isdisjoint(aliases.viewkeys())
    if figure_id_or_alias in aliases:
        figure_id = aliases[figure_id_or_alias]
        figure_alias = figure_id_or_alias
    elif figure_id_or_alias in reverse_aliases:
        figure_alias = reverse_aliases[figure_id_or_alias]
        figure_id = figure_id_or_alias
    else:
        figure_id = figure_alias = figure_id_or_alias
    fignum, figlet = split_figure_id(figure_id_or_alias)
    fig = Figure(id = figure_id, alias = figure_alias, number = fignum, letter = figlet)
    return fig


def split_figure_id(fig_id):
    split = np.argmax([s.isdigit() for s in fig_id])
    fig_number = int(fig_id[:split+1])
    fig_letter = fig_id[split+1:]
    return fig_number, fig_letter


def demo_create_figure(which_figure, live_plots = False, test_mode = False):

    fig = lookup_figure(which_figure)

    params = lambda: None  # Cheap container, like a plastic bag.

    params.dataset_name = \
        'syn_log_reg' if 1 <= fig.number <= 5 else \
        'mnist_ensemble' if fig.number == 6 else \
        bad_value(fig.number)

    if params.dataset_name == 'syn_log_reg':
        params.noise_factor = \
            0.1 if fig.id == 1 else \
            {'A': 0.1, 'B': 0.1, 'C': 0.0, 'D': 0.1}[fig.letter] if fig.number == 2 else \
            0.0

        params.max_test_samples = 100
    elif params.dataset_name == 'mnist_ensemble':
        params.max_training_samples = \
            100 if fig.alias == 'mnist-ensemble' else \
            1000 if fig.alias == 'mnist-ensemble-full' else \
            bad_value(fig.alias)
    else:
        params.max_training_samples = None
        params.max_test_samples = None

    params.max_training_samples = \
        1000 if fig.number in (1, 2) else \
        50 if fig.number in (3, 4) else \
        {'A': 50, 'B': 50, 'C': 1000, 'D': 1000}[fig.letter] if fig.number == 5 else \
        {'A': 100, 'B': 1000}[fig.letter] if fig.number == 6 else \
        bad_value(fig.number)

    params.predictors = {
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
        '6A': ['gibbs', 'herded'],
        '6B': ['gibbs', 'herded'],
        }[fig.id]

    demo_create_figure_from_commands(live_plot = live_plots, test_mode = test_mode, **params.__dict__)


def cascading_defaults(default_dict, kwarg_dict):
    parameter_dict = default_dict.copy()
    remaining_kwargs = kwarg_dict.copy()
    for k in default_dict:
        if k in kwarg_dict:
            default_dict[k] = kwarg_dict[k]
            del remaining_kwargs[k]
    return parameter_dict, remaining_kwargs


def demo_create_figure_from_commands(dataset_name, max_training_samples = None, max_test_samples = None, predictors = ('gibbs', 'herded'),
        w_range = (0, 1), n_steps = 1000, evaluation_fcn = 'mse', live_plot = False, test_mode = False, **kwargs):

    if dataset_name == 'syn_log_reg':
        params, kwargs = cascading_defaults(dict(noise_factor= 0.1, n_dims = 20), kwargs)
        assert max_training_samples is not None and max_test_samples is not None
        dataset = get_logistic_regression_dataset(n_training = max_training_samples, n_test = max_test_samples,
            noise_factor = params['noise_factor'])
        title = 'Logistic Regression Dataset. \nn_training=%s, n_test=%s, n_dims=%s, noise_factor=%s' \
            % (dataset.training_set.n_samples, dataset.test_set.n_samples, dataset.input_shape, params['noise_factor'])
        offline_predictors = {'Optimal': lambda: MockPredictor(lambda x: sigm(x.dot(dataset.w_true)))}
    elif dataset_name == 'mnist_ensemble':
        params, kwargs = cascading_defaults(dict(n_trees = 10), kwargs)
        dataset = get_mnist_rf_ensemble_dataset(
            max_training_samples = max_training_samples,
            max_test_samples = None,
            n_trees = params['n_trees'],
            seed = None
            )
        title = 'MNIST Ensemble using %s training samples' % (max_training_samples, )
        offline_predictors = {'Mode Combination': lambda: MockModePredictor(n_classes = 10)}
    else:
        bad_value(dataset_name)

    n_dim_in=dataset.input_shape[0]
    n_dim_out=dataset.target_shape[0]

    incremental_predictors = get_named_predictors(predictors, n_dim_in, n_dim_out, sample_y = False, w_range = w_range)

    assert len(kwargs)==0, 'Unused kwargs remain: %s' % (kwargs, )

    demo_plot_binary_regression_learning(
        dataset=dataset,
        offline_predictors=offline_predictors,
        incremental_predictors=incremental_predictors,
        test_epochs = np.arange(n_steps).astype(float),
        live_plot = live_plot,
        evaluation_fcn = evaluation_fcn,
        test_mode = test_mode,
        title = title
        )


def demo_plot_binary_regression_learning(dataset, offline_predictors, incremental_predictors, test_epochs,
        live_plot = False, evaluation_fcn = 'mse', test_mode = False, title = 'Results'):
    """
    Code for creating plots in our report.
    """

    learning_curves = compare_predictors(
        dataset = dataset,
        offline_predictor_constructors=offline_predictors,
        incremental_predictor_constructors = incremental_predictors,
        test_points = test_epochs,
        evaluation_function = evaluation_fcn,
        report_test_scores=False,
        on_construction_callback=setup_visualization if live_plot else None
        )

    plot_learning_curves(learning_curves, title = title, xscale = 'sqrt', yscale = 'linear', hang = not test_mode)


if __name__ == '__main__':

    # -- Params -- #
    SPECITY_AS = 'DIRECT'
    TEST_MODE = False
    LIVE_PLOT = False
    # ------------ #

    if SPECITY_AS=='FIGURE':
        # -- Params -- #
        WHICH_FIGURE = '6A'
        # ------------ #
        demo_create_figure(WHICH_FIGURE, live_plot=LIVE_PLOT, test_mode=TEST_MODE)
    elif SPECITY_AS == 'COMMANDS':
        demo_create_figure_from_commands(
            dataset_name = 'mnist_ensemble',
            max_training_samples = None,
            max_test_samples = None,
            predictors = ('gibbs', 'herded'),
            w_range = (0, 1),
            n_steps = 1000,
            evaluation_fcn = 'mse',
            live_plot = LIVE_PLOT,
            test_mode = TEST_MODE,
            )
    elif SPECITY_AS == 'DIRECT':
        # -- Params -- #
        DATASET = get_mnist_rf_ensemble_dataset(max_training_samples=100, max_test_samples = None, n_trees = 50)
        PREDICTOR_FACTORY_FACTORY = get_predictor_factory_factory(n_dim_in=DATASET.input_shape[0], n_dim_out=DATASET.target_shape[0])
        demo_plot_binary_regression_learning(
            dataset=DATASET,
            offline_predictors={'Mode Combination': lambda: MockModePredictor(n_classes = 10)},
            incremental_predictors={
                'gibbs': PREDICTOR_FACTORY_FACTORY(
                    sampling_type = 'gibbs',
                    possible_ws=np.linspace(0, 1, 5),
                    n_alpha = 5
                    ),
                'herded-gibbs': PREDICTOR_FACTORY_FACTORY(
                    sampling_type = 'herded',
                    possible_ws=np.linspace(0, 1, 5),
                    n_alpha = 5
                    ),
                },
            test_epochs = np.arange(2000).astype(float),
            evaluation_fcn='percent_argmax_correct',
            live_plot=LIVE_PLOT,
            test_mode=TEST_MODE,
            title = 'Somethignsomething'
            )
    else:
        bad_value(SPECITY_AS)
