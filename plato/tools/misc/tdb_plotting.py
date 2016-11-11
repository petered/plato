from contextlib import contextmanager
from functools import partial
from plato.interfaces.decorators import tdb_trace, get_tdb_traces
from artemis.plotting.db_plotting import dbplot

__author__ = 'peter'

"""
Special debug plotter that can handle theano variables.
"""

# _UPDATE_CALLBACK_ADDED = False

_CONSTRUCTORS = {}

_tdb_plot_every = None


name_counts = {}


def tdbplot(var, name = None, plot_type = None, draw_every=None, **kwargs):
    """
    Debug plot which can handle theano variables.

    :param data: A theano variable
    :param name: The name of this plot (make it unique from other instances where
        dbplot is called)
    :param kwargs: Passed down to LivePlot.  Some noteable ones:
        plot_mode: {'live', 'static'} (default live).  Determines what kind of plots
            will be made given the data.  "live" tends to make streaming plots, which
            make mores sense when you're running and monitoring.  "static" makes static
            plots, which make more sence for step-by-step debugging.
    """

    if name is None:
        name = '%s-%s' % (str(var), hex(id(var)))
    elif '%c' in name:
        name_counts[name] = 0 if name not in name_counts else name_counts[name] + 1
        num = 0 if name not in name_counts else name_counts[name]
        name = name.replace('%c', str(num))

    if draw_every is None:
        draw_every = _tdb_plot_every

    if plot_type is not None:
        _CONSTRUCTORS[name] = plot_type if isinstance (plot_type, basestring) else (lambda: plot_type)
        # Following is a kludge - the data is flattened in LivePlot, so we reference
        # it by the "flattened" key.
        # get_dbplot_stream().add_plot_type("['%s']" % name, plot_type=plot_type)

    tdb_trace(var, name, callback=partial(set_plot_data_and_update, name=name, draw_every=draw_every))


@contextmanager
def use_tdbplot_period(draw_every):
    global _tdb_plot_every
    old_value = _tdb_plot_every
    _tdb_plot_every = draw_every
    yield
    _tdb_plot_every = old_value


def set_plot_data_and_update(name, draw_every=None):
    data = get_tdb_traces()[name]
    dbplot(data, name, plot_type=_CONSTRUCTORS[name] if name in _CONSTRUCTORS else None, draw_every=draw_every)


    # PLOT_DATA.update(get_tdb_traces())
    # stream = get_dbplot_stream(**kwargs)
    # stream.update()
