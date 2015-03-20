from datetime import datetime
import sys
import os
from matplotlib import pyplot as plt

__author__ = 'peter'


_ORIGINAL_SHOW_FCN = plt.show

_SAVED_FIGURES = []


def get_relative_figures_subdir():
    return 'figures'


def get_local_figures_dir(subdir = None):
    this_path, _ = os.path.split(os.path.abspath(__file__))
    figures_dir = os.path.abspath(os.path.join(sys.executable, '..', '..', '..', get_relative_figures_subdir()))

    if subdir is not None:
        figures_dir = os.path.join(figures_dir, subdir)

    return figures_dir


def save_and_show(fig = None, name = '%T-unnamed_figure', ext = 'pdf', base_dir = get_local_figures_dir(),
        subdir = 'dump', block = None, print_loc = True, show = True):
    """
    Save and show a figure.
    :param fig: The figure in question (or None to just get the current figure)
    :param name: The base-name to save it under.  If you put %T in, it will be replaced with the date/time.  E.g.
        '%T-test_fig.pdf' becomes '2015.03.20T13.35.00.068260-test_fig.pdf'
    :param ext: The file format, if not specified in the name\
    :param base_dir: The root directory for figures
    :param subdir: The subdirectory in which to save this figure.  You can also put %T in this to replace with current time.
    :param block: Should the desiplay block execution?  (If None, just go with current interactivemode setting)
    :param print_loc: Print the save location?
    :param show: Actually show the figure?
    :return: The local file-path of the saved figure.
    """

    if fig is None:
        fig = plt.gcf()

    now = datetime.now().isoformat().replace(':', '.').replace('-', '.')
    subdir = subdir.replace('%T', now)
    name = name.replace('%T', now) + '.'+ext

    is_interactive = plt.isinteractive()
    if block is None:
        block = not is_interactive

    # base, ext = os.path.splitext(name)
    # if ext == '':
    #     ext = default_extension
    full_figure_dir = os.path.join(base_dir, subdir)
    full_figure_loc = os.path.join(full_figure_dir, name)

    try:
        os.makedirs(full_figure_dir)
    except OSError:
        pass

    fig.savefig(full_figure_loc)
    if print_loc:
        print 'Saved figure to "%s"' % (full_figure_loc, )
    _SAVED_FIGURES.append(full_figure_loc)  # Which is technically a memory leak, but you'd have to make a lot of figures.

    if show:
        plt.interactive(not block)
        _ORIGINAL_SHOW_FCN()  # There's an argument block, but it's not supported for all backends
        plt.interactive(is_interactive)

    return full_figure_loc


def get_saved_figure_locs():
    return _SAVED_FIGURES


def get_local_figures_dir(subdir = None):
    this_path, _ = os.path.split(os.path.abspath(__file__))
    figures_dir = os.path.abspath(os.path.join(sys.executable, '..', '..', '..', get_relative_figures_subdir()))

    if subdir is not None:
        figures_dir = os.path.join(figures_dir, subdir)

    try:
        os.makedirs(figures_dir)
    except OSError:
        pass

    return figures_dir


def set_show_callback(cb):
    """
    :param cb: Can be:
        - A function which is called instead of plt.show()
        - None, which just returns things to the default.
    """
    plt.show = cb if cb is not None else _ORIGINAL_SHOW_FCN


def always_save_figures(state = True, **save_and_show_args):
    """
    :param state: True to save figures, False to not save them.
    :param **save_and_show_args: See "save_and_show"
    """

    if state:
        set_show_callback(lambda fig = None: save_and_show(fig, **save_and_show_args))
    else:
        set_show_callback(None)

