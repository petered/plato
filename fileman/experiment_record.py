from collections import OrderedDict
from datetime import datetime
from general.test_mode import is_test_mode
import os
import pickle
from IPython.core.display import display, HTML
from fileman.local_dir import format_filename, make_file_dir, get_relative_path, get_local_path
from fileman.notebook_plots import show_embedded_figure
from fileman.notebook_utils import get_relative_link_from_local_path
from fileman.persistent_print import capture_print
from fileman.saving_plots import clear_saved_figure_locs, get_saved_figure_locs, FigureCollector, \
    set_show_callback, always_save_figures
import matplotlib.pyplot as plt

__author__ = 'peter'


class ExperimentRecord(object):
    """
    Captures all logs and figures generated, and saves the result.  Usage:

    with Experiment() as exp_1:
        do_stuff()
        plot_figures()

    exp_1.show_all_figures()
    """
    VERSION = 0  # We keep this in case we want to change this class, and need record the fact that it is an old version
    # when unpickling.

    def __init__(self, name = 'unnamed', filename = '%T-%N', print_to_console = False, save_result = None, show_figs = None):
        """
        :param name: Base-name of the experiment
        :param filename: Format of the filename (placeholders: %T is replaced by time, %N by name)
        :param experiment_dir: Relative directory (relative to data dir) to save this experiment when it closes
        :param print_to_console: If True, print statements still go to console - if False, they're just rerouted to file.
        :param show_figs: Show figures when the experiment produces them.  Can be:
            'hang': Show and hang
            'draw': Show but keep on going
            False: Don't show figures
            None: 'draw' if in test mode, else 'hang'
        """
        now = datetime.now()
        if save_result is None:
            save_result = not is_test_mode()

        if show_figs is None:
            show_figs = 'draw' if is_test_mode() else 'hang'

        assert show_figs in ('hang', 'draw', False)

        self._experiment_identifier = format_filename(file_string = filename, base_name=name, current_time = now)
        self._experiment_file_path = get_local_experiment_path(self._experiment_identifier)
        self._log_file_name = format_filename('%T-%N', base_name = name, current_time = now)
        self._has_run = False
        self._print_to_console = print_to_console
        self._save_result = save_result
        self._show_figs = show_figs

    def __enter__(self):
        clear_saved_figure_locs()
        if self._show_figs == 'draw':
            plt.ion()
        else:
            plt.ioff()
        self._log_file_path = capture_print(True, to_file = True, log_file_path = self._log_file_name, print_to_console = self._print_to_console)
        always_save_figures(show = self._show_figs, print_loc = False)
        return self

    def __exit__(self, *args):
        # On exit, we read the log file.  After this, the log file is no longer associated with the experiment.
        capture_print(False)

        with open(self._log_file_path) as f:
            self._captured_logs = f.read()

        set_show_callback(None)
        self._captured_figure_locs = get_saved_figure_locs()

        self._has_run = True

        if self._save_result:
            make_file_dir(self._experiment_file_path)
            with open(self._experiment_file_path, 'w') as f:
                pickle.dump(self, f)
                print 'Saving Experiment "%s" at "%s"' % (self._experiment_identifier, self._experiment_file_path)

    def get_identifier(self):
        path = get_relative_path(self.get_file_path(), base_path=get_local_path('experiments'))
        assert path.endswith('.exp.pkl')
        identifier = path[:-len('.exp.pkl')]
        return identifier

    def get_file_path(self):
        return self._experiment_file_path

    def get_logs(self):
        return self._captured_logs

    def get_figure_locs(self):
        return self._captured_figure_locs

    def show_figures(self):
        for loc in self._captured_figure_locs:
            rel_loc = get_relative_link_from_local_path(loc)
            show_embedded_figure(rel_loc)

    def show(self):
        print 'Experiment %s' % (self._experiment_identifier)
        display(HTML("<a href = '%s' target='_blank'>View Log File for this experiment</a>"
                     % get_relative_link_from_local_path(self._log_file_path)))
        self.show_figures()

    def end_and_show(self):
        if not self._has_run:
            self.__exit__()
        self.show()


_CURRENT_EXPERIMENT = None


def start_experiment(*args, **kwargs):
    exp = ExperimentRecord(*args, **kwargs)
    exp.__enter__()
    return exp


def run_experiment(name, exp_dict, print_to_console = True, show_figs = None, **experiment_record_kwargs):
    """
    Run an experiment and save the results.  Return a string which uniquely identifies the experiment.
    You can run the experiment agin later by calling show_experiment(location_string):

    :param name: The name for the experiment (must reference something in exp_dict)
    :param exp_dict: A dict<str:func> where funcs is a function with no arguments that run the experiment.
    :param experiment_record_kwargs: Passed to ExperimentRecord.

    :return: A location_string, uniquely identifying the experiment.
    """

    if isinstance(exp_dict, dict):
        assert name in exp_dict, 'Could not find experiment "%s" in the experiment dictionary with keys %s' % (name, exp_dict.keys())
        func = exp_dict[name]
    else:
        assert hasattr(exp_dict, '__call__')
        func = exp_dict

    with ExperimentRecord(name = name, print_to_console=print_to_console, show_figs=show_figs, **experiment_record_kwargs) as exp_rec:
        func()

    return exp_rec.get_identifier()


def run_notebook_experiment(name, exp_dict, **experiment_record_kwargs):
    """
    Run an experiment with settings more suited to an IPython notebook.  Here, we want to redirect all
    output to a log file, and not show the figures immediately.
    """
    run_experiment(name, exp_dict, print_to_console = False, show_figs = False, **experiment_record_kwargs)


def get_local_experiment_path(identifier):
    return format_filename(identifier, directory = get_local_path('experiments'), ext = 'exp.pkl')


def show_experiment(identifier):
    """
    Show the results of an experiment (plots and logs)
    :param identifier: A string uniquely identifying the experiment
    """
    local_path = get_local_experiment_path(identifier)
    assert os.path.exists(local_path), "Couldn't find experiment '%s' at '%s'" % (identifier, local_path)
    with open(local_path) as f:
        exp_rec = pickle.load(f)
    exp_rec.show()


def merge_experiment_dicts(*dicts):
    """
    Merge dictionaries of experiments, checking that names are unique.
    """
    merge_dict = OrderedDict()
    for d in dicts:
        assert not any(k in merge_dict for k in d), "Experiments %s has been defined twice." % ([k for k in d.keys() if k in merge_dict],)
        merge_dict.update(d)
    return merge_dict
