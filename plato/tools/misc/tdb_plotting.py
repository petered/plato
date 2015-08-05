from plato.interfaces.decorators import tdb_trace, get_tdb_traces
from plotting.db_plotting import get_dbplot_stream, set_plot_data_and_update, PLOT_DATA

__author__ = 'peter'

"""
Special debug plotter that can handle theano variables.
"""

_UPDATE_CALLBACK_ADDED = False


def tdbplot(var, name = None, plot_type = None, **kwargs):
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

    global _UPDATE_CALLBACK_ADDED
    if not _UPDATE_CALLBACK_ADDED:
        callback = lambda: set_plot_data_and_update(**kwargs)
        _UPDATE_CALLBACK_ADDED = True
    else:
        callback = None
    if plot_type is not None:
        # Following is a kludge - the data is flattened in LivePlot, so we reference
        # it by the "flattened" key.
        get_dbplot_stream().add_plot_type("['%s']" % name, plot_type=plot_type)
    tdb_trace(var, name, callback=callback)


def set_plot_data_and_update(**kwargs):
    PLOT_DATA.update(get_tdb_traces())
    stream = get_dbplot_stream(**kwargs)
    stream.update()
