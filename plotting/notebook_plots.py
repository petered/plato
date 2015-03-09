import os
from datetime import datetime
from matplotlib import pyplot as plt
from IPython.display import FileLink
__author__ = 'peter'


def get_local_figures_dir():
    this_path, _ = os.path.split(os.path.abspath(__file__))
    figures_dir = os.path.abspath(os.path.join(this_path, '..', get_relative_figures_dir()))

    try:
        os.makedirs(figures_dir)
    except OSError:
        pass

    return figures_dir


def get_relative_figures_dir():
    return 'figures'


class FigureLink(FileLink):

    def _repr_html_(self):
        """Just skip the check for file existing
        """
        _, filename = os.path.split(self.path)
        return "See <a href='%s' target='_blank'>this figure</a>.  See <a href='%s' target='_blank'>all figures</a>" % (self.path, '/tree/figures')


def link_and_show(name = 'unnamed_figure', default_extension = 'pdf', preprend_datetime = True):
    """
    Use this function to show a plot in IPython Notebook, and provide a link to download the figure.

    :param name: The figure name.  The extension, if included, specifies the file type.
    :param default_extension: The default extension to use.
    :param preprend_datetime: Prepend the datetime to the filename.  This makes it so new plots don't overwrite old ones
        with the same name.
    :return: A FigureLink.  In IPython notebook, this will display as a link to the figure (and to the figures folder)
    """
    base, ext = os.path.splitext(name)
    if ext == '':
        ext = default_extension
        name = base+'.'+ext

    name = datetime.now().isoformat().replace(':', '.').replace('-', '.')+'_'+name if preprend_datetime else name

    full_figure_loc = os.path.join(get_local_figures_dir(), name)
    relative_path = os.path.join('/files', get_relative_figures_dir(), name)
    plt.savefig(full_figure_loc)
    return FigureLink(relative_path)
