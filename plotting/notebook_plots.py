from IPython.core.display import HTML
import os
from IPython.display import display
from plotting.saving_plots import save_and_show, get_local_figures_dir, set_show_callback, get_relative_figures_subdir

__author__ = 'peter'


def always_link_figures(state = True, **link_and_show_arg):

    set_show_callback(lambda fig = None: link_and_show(fig=fig, **link_and_show_arg) if state else None)


def link_and_show(**save_and_show_kwargs):
    """
    Use this function to show a plot in IPython Notebook, and provide a link to download the figure.

    :param name: The figure name.  The extension, if included, specifies the file type.
    :param default_extension: The default extension to use.
    :param preprend_datetime: Prepend the datetime to the filename.  This makes it so new plots don't overwrite old ones
        with the same name.
    :return: A FigureLink.  In IPython notebook, this will display as a link to the figure (and to the figures folder)
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

