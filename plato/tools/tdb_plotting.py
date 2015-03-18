from general.nested_structures import flatten_struct
from plotting.db_plotting import dbplot
from theano.compile.sharedvalue import SharedVariable
from theano.tensor.var import TensorVariable

__author__ = 'peter'

"""
Special debug plotter that can handle theano variables.
"""


def tdbplot(data, name, **kwargs):
    """
    Debug plot, which can handle theano variables (as long as they have test
    values attached).

    :param data: Any data structure containing your data/tensors
    :param name: The name of this plot (make it unique from other instances where
        dbplot is called)
    :param kwargs: Passed down to LivePlot.  Some noteable ones:
        plot_mode: {'live', 'static'} (default live).  Determines what kind of plots
            will be made given the data.  "live" tends to make streaming plots, which
            make mores sense when you're running and monitoring.  "static" makes static
            plots, which make more sence for step-by-step debugging.
    """
    # TODO: Add test/demo of this, because it's pretty cool

    custom_handlers = {
        TensorVariable: lambda d: d.tag.test_value if hasattr(d, 'tag') else '<No test value>',
        SharedVariable: lambda d: d.get_value(),
    }

    return dbplot(data, name=name, custom_handlers=custom_handlers, **kwargs)
