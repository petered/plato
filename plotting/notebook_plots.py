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


def link_and_show(name = 'unnamed_figure.pdf', default_extension = 'pdf', preprend_datetime = True):

    base, ext = os.path.splitext(name)
    if ext == '':
        ext = default_extension
        name = base+'.'+ext

    name = datetime.now().isoformat().replace(':', '.').replace('-', '.')+'_'+name if preprend_datetime else name

    full_figure_loc = os.path.join(get_local_figures_dir(), name)
    relative_path = os.path.join('/files', get_relative_figures_dir(), name)
    plt.savefig(full_figure_loc)
    return FigureLink(relative_path)
