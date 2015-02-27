from collections import OrderedDict, namedtuple
from experimental.rf_ensembles import MockModePredictor, get_mnist_rf_ensemble_dataset
import numpy as np
from plotting.live_plotting import LiveStream
from utils.benchmarks.compare_predictors import compare_predictors
from utils.benchmarks.plot_learning_curves import plot_learning_curves
from utils.datasets.crohns_disease import get_crohns_dataset
from utils.datasets.synthetic_logistic import get_logistic_regression_dataset
from utils.predictors.mock_predictor import MockPredictor
from utils.tools.mymath import sigm
from plato.tools.sampling import GibbsRegressor, HerdedGibbsRegressor, SamplingPredictor


Figure = namedtuple('Figure', ['id', 'number', 'letter', 'alias'])


__author__ = 'peter'

"""
Here, we do logistic regression with binary weights.
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


def _get_dataset_for_figures_1_thru_5(fig):

    assert 1 <= fig.number <= 5

    n_dims = 20

    noise_factor = \
        0.1 if fig.id == 1 else \
        {'A': 0.1, 'B': 0.1, 'C': 0.0, 'D': 0.1}[fig.letter] if fig.number == 2 else \
        0.0

    n_training = \
        1000 if fig.number in (1, 2) else \
        50 if fig.number in (3, 4) else \
        {'A': 50, 'B': 50, 'C': 1000, 'D': 1000}[fig.letter] if fig.number == 5 else \
        500

    n_test = 100

    dataset = get_logistic_regression_dataset(n_dims = n_dims,
        n_training=n_training, n_test=n_test, noise_factor = noise_factor)

    title = 'Logistic Regression Dataset. \nn_training=%s, n_test=%s, n_dims=%s, noise_factor=%s' \
        % (dataset.training_set.n_samples, dataset.test_set.n_samples, dataset.input_shape, noise_factor)

    return dataset, title


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


def get_predictors_for_figure(which_figure, n_dim_in, n_dim_out, w_range = (0, 1)):

    sample_y = True if which_figure in ('2G', '2H') else False,

    predictor_for_figure = {
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
        }[which_figure]

    return get_named_predictors(predictor_for_figure, n_dim_in, n_dim_out, sample_y=sample_y, w_range = w_range)


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


def _get_dataset_for_figure_6(fig):

    n_trees = 10

    max_training_samples = \
        100 if fig.alias == 'mnist-ensemble' else \
        1000 if fig.alias == 'mnist-ensemble-full' else \
        bad_value(fig.alias)

    dataset = get_mnist_rf_ensemble_dataset(
        max_training_samples = max_training_samples,
        max_test_samples = None,
        n_trees = n_trees,
        seed = None
        )

    return dataset, 'MNIST Ensemble with %s Trees, %s training_samples, %s test samples' \
           % (n_trees, dataset.training_set.n_samples, dataset.test_set.n_samples)


def demo_create_figure(which_figure, live_plot = False, test_mode = False):
    """
    :param which_figure: Which figure of the report to replicate.  Or "X" for just
        experimenting with stuff.  The report is here:
        https://docs.google.com/document/d/1zBQdI-1tcEvqmizCuL2GX_ceqhLnOvGoN8ljciw9uGE/edit?usp=sharing
    """

    fig = lookup_figure(which_figure)

    dataset, title = \
        _get_dataset_for_figures_1_thru_5(fig) if 1 <= fig.number <= 5 else \
        _get_dataset_for_figure_6(fig) if fig.number == 6 else \
        bad_value(fig.number)

    evaluation_fcn = \
        'mse' if 1 <= fig.number <= 5 else \
        'percent_argmax_correct' if fig.number == 6 else \
        bad_value(fig.number)

    offline_predictors = \
        {'Optimal': lambda: MockPredictor(lambda x: sigm(x.dot(dataset.w_true)))} if 1 <= fig.number <= 5 else \
        {'Mode Combination': lambda: MockModePredictor(n_classes = 10)} if fig.number == 6 else \
        bad_value(fig.number)

    incremental_predictors = get_predictors_for_figure(fig.id, n_dim_in=dataset.input_shape[0], n_dim_out = dataset.target_shape[0])

    n_steps = \
        3 if test_mode else \
        1000 if 1 <= fig.number <= 4 else \
        10000 if fig.number == 5 else \
        2000 if fig.number == 6 else \
        bad_value(fig.number)

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
    CREATE_FIGURE = True
    TEST_MODE = False
    LIVE_PLOT = False
    # ------------ #

    if CREATE_FIGURE:
        # -- Params -- #
        WHICH_FIGURE = '6A'
        # ------------ #
        demo_create_figure(WHICH_FIGURE, live_plot=LIVE_PLOT, test_mode=TEST_MODE)
    else:
        # -- Params -- #
        DATASET = get_crohns_dataset()
        PREDICTORS = ['gibbs-1/4', 'herded-1/4']
        EVALUATION_FCN = 'percent_argmax_correct'
        N_TREES = 10
        N_STEPS = 2000
        # ------------ #
        offline_predictors = {'Mode Combination': lambda: MockModePredictor(n_classes = 10)}
        incremental_predictors = get_named_predictors(PREDICTORS, DATASET.input_shape[0], DATASET.target_shape[0])
        predictor_factory_factory = get_predictor_factory_factory(n_dim_in=DATASET.input_shape[0], n_dim_out=DATASET.target_shape[0])
        demo_plot_binary_regression_learning(
            dataset=DATASET,
            offline_predictors=offline_predictors,
            incremental_predictors={
                'gibbs': predictor_factory_factory(
                    sampling_type = 'gibbs',
                    possible_ws=np.linspace(0, 1, 5),
                    n_alpha = 5
                    ),
                'herded-gibbs': predictor_factory_factory(
                    sampling_type = 'herded',
                    possible_ws=np.linspace(0, 1, 5),
                    n_alpha = 5
                    ),
                },
            test_epochs = np.arange(N_STEPS).astype(float),
            live_plot=LIVE_PLOT,
            evaluation_fcn=EVALUATION_FCN,
            test_mode=TEST_MODE,
            title = 'SADFSDfdS'
            )
