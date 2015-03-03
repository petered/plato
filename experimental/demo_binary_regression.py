from collections import OrderedDict
from experimental.rf_ensembles import MockModePredictor, get_mnist_rf_ensemble_dataset
from general.kwarg_dealer import KwargDealer
from general.redict import ReDict, ReCurseDict
import numpy as np
from plotting.live_plotting import LiveStream
from utils.benchmarks.compare_predictors import compare_predictors
from utils.benchmarks.plot_learning_curves import plot_learning_curves
from utils.datasets.crohns_disease import get_crohns_dataset
from utils.datasets.synthetic_logistic import get_logistic_regression_dataset
from utils.predictors.bad_predictors import DistributionPredictor, MockPredictor
from utils.tools.mymath import sigm
from plato.tools.sampling import GibbsRegressor, HerdedGibbsRegressor, SamplingPredictor
from utils.tools.processors import OneHotEncoding
from functools import partial


__author__ = 'peter'

"""
Here, we do logistic regression with binary weights.

This is an exercize in organization.  There are 3 levels at which we can define our experment:

1) Just say the figure number
2) Give high-level "user" commands
3) The actual predictors and datasets to use.

1 should translate to 2 should translate to 3.
"""


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


def get_predictor_factory(n_dim_in, n_dim_out, sample_y, sampling_type, n_alpha, alpha_update_policy = 'sequential', possible_ws = (0, 1)):
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


def get_named_predictors(names, n_dim_in, n_dim_out, sample_y = False, w_range = (0, 1)):
    
    predictor_factory = partial(get_predictor_factory, n_dim_in=n_dim_in, n_dim_out=n_dim_out, sample_y=sample_y)

    w_min, w_max = w_range

    full_set_of_predictors = {
        'gibbs': predictor_factory(sampling_type = 'gibbs', n_alpha = 1, alpha_update_policy='sequential', possible_ws = (w_min, w_max)),
        'herded': predictor_factory(sampling_type = 'herded', n_alpha = 1, alpha_update_policy='sequential', possible_ws = (w_min, w_max)),
        'gibbs-rand': predictor_factory(sampling_type = 'gibbs', n_alpha = 1, alpha_update_policy='random', possible_ws = (w_min, w_max)),
        'herded-rand': predictor_factory(sampling_type = 'herded', n_alpha = 1, alpha_update_policy='random', possible_ws = (w_min, w_max)),
        'gibbs-1/4': predictor_factory(sampling_type = 'gibbs', n_alpha = n_dim_in/4, alpha_update_policy='sequential', possible_ws = (w_min, w_max)),
        'herded-1/4': predictor_factory(sampling_type = 'herded', n_alpha = n_dim_in/4, alpha_update_policy='sequential', possible_ws = (w_min, w_max)),
        'gibbs-1/4-rand': predictor_factory(sampling_type = 'gibbs', n_alpha = n_dim_in/4, alpha_update_policy='random', possible_ws = (w_min, w_max)),
        'herded-1/4-rand': predictor_factory(sampling_type = 'herded', n_alpha = n_dim_in/4, alpha_update_policy='random', possible_ws = (w_min, w_max)),
        'gibbs-1/2': predictor_factory(sampling_type = 'gibbs', n_alpha = n_dim_in/2, alpha_update_policy='sequential', possible_ws = (w_min, w_max)),
        'herded-1/2': predictor_factory(sampling_type = 'herded', n_alpha = n_dim_in/2, alpha_update_policy='sequential', possible_ws = (w_min, w_max)),
        'gibbs-1/2-rand': predictor_factory(sampling_type = 'gibbs', n_alpha = n_dim_in/2, alpha_update_policy='random', possible_ws = (w_min, w_max)),
        'herded-1/2-rand': predictor_factory(sampling_type = 'herded', n_alpha = n_dim_in/2, alpha_update_policy='random', possible_ws = (w_min, w_max)),
        'gibbs-full': predictor_factory(sampling_type = 'gibbs', n_alpha = n_dim_in, alpha_update_policy='sequential', possible_ws = (w_min, w_max)),
        'herded-full': predictor_factory(sampling_type = 'herded', n_alpha = n_dim_in, alpha_update_policy='sequential', possible_ws = (w_min, w_max)),
        'gibbs-full-rand': predictor_factory(sampling_type = 'gibbs', n_alpha = n_dim_in, alpha_update_policy='random', possible_ws = (w_min, w_max)),
        'herded-full-rand': predictor_factory(sampling_type = 'herded', n_alpha = n_dim_in, alpha_update_policy='random', possible_ws = (w_min, w_max)),
        'gibbs-5choice': predictor_factory(sampling_type = 'gibbs', n_alpha = 1, alpha_update_policy='sequential', possible_ws=np.linspace(w_min, w_max, 5)),
        'herded-5choice': predictor_factory(sampling_type = 'herded', n_alpha = 1, alpha_update_policy='sequential', possible_ws=np.linspace(w_min, w_max, 5)),
        'gibbs-20choice': predictor_factory(sampling_type = 'gibbs', n_alpha = 1, alpha_update_policy='sequential', possible_ws=np.linspace(w_min, w_max, 20)),
        'herded-20choice': predictor_factory(sampling_type = 'herded', n_alpha = 1, alpha_update_policy='sequential', possible_ws=np.linspace(w_min, w_max, 20)),
        }  # Could factor this out into a ReCurseDict, but there is a limit...
    
    predictors_to_compare = OrderedDict((k, full_set_of_predictors[k]) for k in names)

    return predictors_to_compare

figure_numbers = {
    '1': 'synlog-simple',
    '2A': 'synlog-comp-baseline_1',
    '2B': 'synlog-comp-baseline_2',
    '2C': 'synlog-comp-noiseless',
    '2D': 'synlog-comp-noisy',
    '2E': 'synlog-comp-10D',
    '2F': 'synlog-comp-5D',
    '2G': 'synlog-comp-40D',
    '2H': 'synlog-comp-10ksamples',
    '2I': 'synlog-comp-sample_y_1',
    '2J': 'synlog-comp-sample_y_2',
    '3A': 'synlog-n_alpha-gibbs',
    '3B': 'synlog-n_alpha-herded',
    '4A': 'synlog-n_alpha-seq',
    '4B': 'synlog-n_alpha-1/4',
    '4C': 'synlog-n_alpha-1/2',
    '4D': 'synlog-n_alpha-full',
    '5A': 'synlog-wvals-gibbs-50samples',
    '5B': 'synlog-wvals-herded-50samples',
    '5C': 'synlog-wvals-gibbs-1ksamples',
    '5D': 'synlog-wvals-herded-1ksamples',
    '6A': 'synlog-nd-lowN-lowD',
    '6B': 'synlog-nd-lowN-highD',
    '6C': 'synlog-nd-highN-lowD',
    '6D': 'synlog-nd-highN-highD',
    '7A': 'mnist_ensemble-binary-100sample',
    '7B': 'mnist_ensemble-5choice-100sample',
    '7C': 'mnist_ensemble-binary-1000sample',
    '7D': 'mnist_ensemble-5choice-1000sample',
    '8A': 'crohns'
    }

figure_names = {fig_num: fig_name for fig_name, fig_num in figure_numbers.iteritems()}

assert set(figure_names.viewkeys()).isdisjoint(figure_numbers.viewkeys()), \
    'There should be no overlab between figure numbers and figure names.'


def get_figure_name(figure_name_or_number):
    """
    Return the name
    """
    if figure_name_or_number in figure_numbers:
        return figure_numbers[figure_name_or_number]
    else:
        assert figure_name_or_number in figure_names, 'There is not figure "%s"' % (figure_name_or_number, )
        return figure_name_or_number


def demo_create_figure(which_figure, live_plot = False, test_mode = False, **overriding_kwargs):

    fig_name = get_figure_name(which_figure)

    params = lambda: None  # Cheap container, like a plastic bag.

    if fig_name.startswith('synlog'):

        params.dataset_name = 'syn_log_reg'

        params.noise_factor = ReDict({
            'synlog-gibbs': 0.1,
            'synlog-comp-noiseless': 0.,
            'synlog-comp-noisy': 1.,
            None: 0.0
            })[fig_name]

        params.max_training_samples = ReCurseDict({
            'synlog-(simple|comp-.*)': 1000,
            'synlog-n_alpha-.*': 50,
            'synlog-wvals-.*': {'.*-50samples': 50, '.*-1ksamples': 1000},
            'synlog-nd-.*': {'.*-lowN-.*': 100, '.*-highN-.*': 1000}
            })[fig_name]

        params.n_dims = ReCurseDict({
            'synlog-comp-.*': {'.*-10D': 10, '.*-5D': 5, '.*-40D': 40, None: 20},
            'synlog-nd-.*': {'.*-lowD': 20, '.*-highD': 500},
            None: 20
            })[fig_name]

        params.max_test_samples = 100

        params.predictors = ReDict({
            'synlog-simple':                                ['gibbs'],
            'synlog-comp-*':                                ['gibbs', 'herded'],
            'synlog-n_alpha-gibbs':                         ['gibbs', 'gibbs-1/4', 'gibbs-1/2', 'gibbs-full'],
            'synlog-n_alpha-herded':                        ['herded', 'herded-1/4', 'herded-1/2', 'herded-full'],
            'synlog-n_alpha-seq':                           ['gibbs', 'gibbs-rand', 'herded', 'herded-rand'],
            'synlog-n_alpha-1/4':                           ['gibbs-1/4', 'gibbs-1/4-rand', 'herded-1/4', 'herded-1/4-rand'],
            'synlog-n_alpha-1/2':                           ['gibbs-1/2', 'gibbs-1/2-rand', 'herded-1/2', 'herded-1/2-rand'],
            'synlog-n_alpha-full':                          ['gibbs-full', 'gibbs-full-rand', 'herded-full', 'herded-full-rand'],
            'synlog-wvals-gibbs-(50samples|1ksamples)':     ['gibbs', 'gibbs-5choice', 'gibbs-20choice'],
            'synlog-wvals-herded-(50samples|1ksamples)':    ['herded', 'herded-5choice', 'herded-20choice'],
            'synlog-nd-.*':                                 ['gibbs', 'herded'],
            })[fig_name]

    elif fig_name.startswith('crohns'):
        params.dataset_name = 'crohns'
        params.predictors = ('gibbs-5choice', 'herded-5choice')
        params.w_range = (-1, 1)
        params.n_steps = 30000
        params.evaluation_fcn = 'percent_argmax_correct'

    elif fig_name.startswith('mnist_ensemble'):

        params.dataset_name = 'mnist_ensemble'
        params.predictors = ReCurseDict({
            '.*-binary-.*': ('gibbs', 'herded'),
            '.*-5choice-.*': ('gibbs-5choice', 'herded-5choice'),
            })[fig_name]
        params.max_training_samples = ReCurseDict({
            '.*-100sample': 100,
            '.*-1000sample': 1000,
            })[fig_name]
        params.w_range = (0, 1)
        params.n_steps = 2000
        params.evaluation_fcn = 'percent_argmax_correct'

    else:
        bad_value(fig_name)

    kd = KwargDealer(overriding_kwargs)  # Allows you to override defaults from the top
    passed_down_kwargs = kd.deal(params.__dict__)
    kd.assert_empty()  # Note, we could not do this and actually allow you to override parameters all the way to the bottom.

    demo_create_figure_from_commands(live_plot = live_plot, test_mode = test_mode, **passed_down_kwargs)


def demo_create_figure_from_commands(dataset_name, max_training_samples = None, max_test_samples = None, predictors = ('gibbs', 'herded'),
        w_range = (0, 1), n_steps = 1000, evaluation_fcn = 'mse', live_plot = False, test_mode = False, **kwargs):

    kd = KwargDealer(kwargs)

    if test_mode:
        max_training_samples = 5
        n_steps = 3

    if dataset_name == 'syn_log_reg':
        params = kd.deal(dict(noise_factor= 0.1, n_dims = 20))
        assert max_training_samples is not None and max_test_samples is not None
        dataset = get_logistic_regression_dataset(n_training = max_training_samples, n_test = max_test_samples,
            n_dims = params['n_dims'], noise_factor = params['noise_factor'])
        title = 'Logistic Regression Dataset. \nn_training=%s, n_test=%s, n_dims=%s, noise_factor=%s' \
            % (dataset.training_set.n_samples, dataset.test_set.n_samples, dataset.input_shape, params['noise_factor'])
        offline_predictors = {'Optimal': lambda: MockPredictor(lambda x: sigm(x.dot(dataset.w_true)))}
    elif dataset_name == 'mnist_ensemble':
        params = kd.deal(dict(n_trees = 10))
        dataset = get_mnist_rf_ensemble_dataset(
            max_training_samples = max_training_samples,
            max_test_samples = None,
            n_trees = params['n_trees'],
            seed = None
            )
        title = 'MNIST Ensemble using %s training samples' % (max_training_samples, )
        offline_predictors = {'Mode Combination': lambda: MockModePredictor(n_classes = 10)}
    elif dataset_name == 'crohns':
        dataset = get_crohns_dataset(frac_training=0.7).process_with(targets_processor = lambda (x, ): (OneHotEncoding()(x), ))
        title = 'Crohns Disease.'
        offline_predictors = {
            'MostFrequent': lambda: DistributionPredictor(),
            }
    else:
        bad_value(dataset_name)

    n_dim_in=dataset.input_shape[0]
    n_dim_out=dataset.target_shape[0]

    incremental_predictors = get_named_predictors(predictors, n_dim_in, n_dim_out, sample_y = False, w_range = w_range)

    kd.assert_empty()

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


if __name__ == '__main__':

    # -- Params -- #
    SPECITY_AS = 'direct'
    # ------------ #

    if SPECITY_AS=='figure':
        # -- Params -- #
        demo_create_figure(
            '7D',
            live_plot=False,
            test_mode=False
        )
        # ------------ #

    elif SPECITY_AS == 'commands':
        demo_create_figure_from_commands(
            dataset_name = 'crohns',
            max_training_samples = None,
            max_test_samples = None,
            predictors = ('gibbs-5choice', 'herded-5choice'),
            w_range = (-1, 1),
            n_steps = 10000,
            evaluation_fcn = 'percent_argmax_correct',
            live_plot = False,
            test_mode = False,
            )
    elif SPECITY_AS == 'direct':
        # -- Params -- #
        DATASET = get_mnist_rf_ensemble_dataset(max_training_samples=100, max_test_samples = None, n_trees = 50)
        PREDICTOR_FACTORY = partial(get_predictor_factory,
            n_dim_in=DATASET.input_shape[0],
            n_dim_out=DATASET.target_shape[0],
            sample_y = False,
            possible_ws=np.linspace(0, 1, 5),
            n_alpha = 5
            )
        demo_plot_binary_regression_learning(
            dataset=DATASET,
            offline_predictors={'Mode Combination': lambda: MockModePredictor(n_classes = 10)},
            incremental_predictors={
                'gibbs': PREDICTOR_FACTORY(sampling_type = 'gibbs'),
                'herded-gibbs': PREDICTOR_FACTORY(sampling_type = 'herded'),
                },
            test_epochs = np.arange(2000).astype(float),
            evaluation_fcn='percent_argmax_correct',
            live_plot=False,
            test_mode=False,
            title = 'Somethignsomething'
            )
    else:
        bad_value(SPECITY_AS)
