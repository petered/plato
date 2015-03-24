from general.nested_structures import flatten_struct
from plato.interfaces.decorators import tdb_trace, get_tdb_traces
from plotting.db_plotting import dbplot, get_dbplot_stream
from theano.compile.sharedvalue import SharedVariable
from theano.tensor.var import TensorVariable

__author__ = 'peter'

"""
Special debug plotter that can handle theano variables.
"""

_UPDATE_CALLBACK_ADDED = False


def tdbplot(var, name, **kwargs):
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
    # TODO: Add test/demo of this, because it's pretty cool

    global _UPDATE_CALLBACK_ADDED
    if not _UPDATE_CALLBACK_ADDED:
        callback = lambda: dbplot(get_tdb_traces(), **kwargs)
        _UPDATE_CALLBACK_ADDED = True
    else:
        callback = None

    tdb_trace(var, name, callback=callback)

    #
    #
    # return dbplot(data, name=name, custom_handlers=custom_handlers, **kwargs)
    #
    #
