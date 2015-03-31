from datetime import datetime
import pickle
from IPython.core.display import display, HTML
from fileman.local_dir import format_filename, make_file_dir
from fileman.notebook_plots import show_embedded_figure
from fileman.notebook_utils import get_relative_link_from_local_path
from fileman.persistent_print import capture_print, PrintAndStoreLogger
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
    VERSION = 0  # We keep this in case we want to change this class.

    def __init__(self, name = 'unnamed', filename = '%T-%N', experiment_dir = 'experiments', print_to_console = False):
        """
        :param name: Base-name of the experiment
        :param filename: Format of the filename (placeholders: %T is replaced by time, %N by name)
        :param experiment_dir: Relative directory (relative to data dir) to save this experiment when it closes
        :param print_to_console: If True, print statements still go to console - if False, they're just rerouted to file.
        """
        now = datetime.now()
        self._experiment_file_path = format_filename(filename, base_name=name, current_time = now, rel_dir = experiment_dir, ext = 'exp.pkl')
        self._log_file_name = format_filename('%T-%N', base_name = name, current_time = now)
        self._has_run = False
        self._print_to_console = print_to_console

    def __enter__(self):
        clear_saved_figure_locs()
        plt.ioff()
        self._log_file_path = capture_print(True, to_file = True, log_file_path = self._log_file_name, print_to_console = False)
        always_save_figures(show = False, print_loc = False)
        return self

    def __exit__(self, *args):
        # On exit, we read the log file.  After this, the log file is no longer associated with the experiment.
        capture_print(False)

        with open(self._log_file_path) as f:
            self._captured_logs = f.read()

        set_show_callback(None)
        self._captured_figure_locs = get_saved_figure_locs()

        self._has_run = True

        make_file_dir(self._experiment_file_path)
        with open(self._experiment_file_path, 'w') as f:
            pickle.dump(self, f)

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

    def end_and_show(self):
        if not self._has_run:
            self.__exit__()
        display(HTML("<a href = '%s' target='_blank'>View Log File for this experiment</a>"
                     % get_relative_link_from_local_path(self._log_file_path)))
        self.show_figures()


_CURRENT_EXPERIMENT = None


def start_experiment(*args, **kwargs):
    exp = ExperimentRecord(*args, **kwargs)
    exp.__enter__()
    return exp
