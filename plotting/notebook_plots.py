from IPython.core.display import HTML
import os
from IPython.display import display
from plotting.saving_plots import save_and_show, get_local_figures_dir, set_show_callback, get_relative_figures_subdir

__author__ = 'peter'


def always_link_figures(state = True, **link_and_show_arg):
    """
    Call this function to always
    :param state: True to display links to plots
    :param show: Set to False if you just want to print the link, and not actually show the plot.
    :param link_and_show_arg: Passed down to link_and_show/save_and_show
    """

    set_show_callback(lambda fig = None: link_and_show(fig=fig, **link_and_show_arg) if state else None)


def link_and_show(**save_and_show_kwargs):
    """
    Use this function to show a plot in IPython Notebook, and provide a link to download the figure.
    See function save_and_show for parameters.
    """

    base_dir = get_local_figures_dir()
    full_figure_loc = save_and_show(print_loc = False, base_dir=base_dir, **save_and_show_kwargs)
    assert full_figure_loc.startswith(base_dir)

    relative_figure_loc = full_figure_loc[len(base_dir):]

    if relative_figure_loc.startswith(os.sep):
        relative_figure_loc = relative_figure_loc[1:]

    relative_path = os.path.join('/files', get_relative_figures_subdir(), relative_figure_loc)

    display(HTML("See <a href='%s' target='_blank'>this figure</a>.  See <a href='%s' target='_blank'>all figures</a>"
            % (relative_path, '/tree/figures')))
