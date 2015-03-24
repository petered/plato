from datetime import datetime
import sys
from StringIO import StringIO

from IPython.core.display import display, HTML
from fileman.local_dir import get_local_path, make_file_dir
from fileman.notebook_utils import get_relative_link_from_local_path, get_relative_link_from_relative_path
import os


__author__ = 'peter'

"""
Save Print statements:

Useful in ipython notebooks where you lose output when printing to the browser.

On advice from:
http://stackoverflow.com/questions/4675728/redirect-stdout-to-a-file-in-python
"""

_ORIGINAL_STDOUT = sys.stdout


def get_local_log_dir(subdir = None):
    figures_dir = get_local_path('logs')
    if subdir is not None:
        figures_dir = os.path.join(figures_dir, subdir)
    return figures_dir


class PrintAndStoreLogger(object):
    def __init__(self, log_file_path = None, base_dir = get_local_log_dir(), print_to_console = True):

        self._print_to_console = print_to_console

        now = datetime.now().isoformat().replace(':', '.').replace('-', '.')
        if log_file_path is not None:
            self._log_file_path = os.path.join(base_dir, log_file_path.replace('%T', now))
            make_file_dir(self._log_file_path)
            self.log = open(self._log_file_path, 'w')
        else:
            self._log_file_path = None
            self.log = StringIO()
        self.terminal = _ORIGINAL_STDOUT

    def get_log_file_path(self):
        return self._log_file_path

    def write(self, message):
        if self._print_to_console:
            self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def read(self):
        if self._log_file_path is None:
            return self.log.getvalue()
        else:
            with open(self._log_file_path) as f:
                txt = f.read()
            return txt

    def __getattr__(self, item):
        return getattr(self.terminal, item)


def capture_print(state = True, to_file = False, log_file_path = 'dump/%T-log.txt', **print_and_store_kwargs):
    """
    :param state:
    :param to_file:
    :param log_file_path:
    :param print_and_store_kwargs:
    :return:
    """

    if state:
        log_file_path = log_file_path if to_file else None
        logger = PrintAndStoreLogger(log_file_path=log_file_path, **print_and_store_kwargs)
        if to_file:
            relative_link = get_relative_link_from_local_path(logger.get_log_file_path())
            log_folder_link = get_relative_link_from_relative_path('logs')
            display(HTML("Writing to <a href='%s' target='_blank'>this log file</a>.  See <a href='%s' target='_blank'>all logs</a>"
                % (relative_link, log_folder_link)))
        sys.stdout = logger
        return logger.get_log_file_path()
    else:
        sys.stdout = _ORIGINAL_STDOUT


def read_print():
    return sys.stdout.read()


def reprint():
    assert isinstance(sys.stdout, PrintAndStoreLogger), "Can't call reprint unless you've turned on capture_print"
    # Need to avoid exponentially growing prints...
    current_stdout = sys.stdout
    sys.stdout = _ORIGINAL_STDOUT
    print read_print()
    sys.stdout = current_stdout
