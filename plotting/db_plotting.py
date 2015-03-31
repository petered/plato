from plotting.live_plotting import LiveStream

__author__ = 'peter'


PLOT_DATA = {}

_SPECIFIED_PLOTS = set()

STREAM = None


def dbplot(data, name = None, **kwargs):
    """
    Quick plot of some variable - you can call this in a loop and it will know to update the
    same plot.
    :param data: The data to plot
    :param name: The name of this plot (you need to specify this if you want to make more than one plot)
    :param plot_mode: Affects the type of plots generated.
        'live' is more appropriate when you're monitoring something and want an online plot with memory
        'static' is better for step-by-step debugging, or plotting from the debugger.
    """

    # global STREAM
    # if STREAM is None:
    #     STREAM = LiveStream(lambda: PLOT_DATA, plot_mode=plot_mode, **kwargs)
    if not isinstance(name, str):
        name = str(name)


    # TODO: Actually properly allow speciying plot types.

    # if any(k not in _SPECIFIED_PLOTS for k in plot_constructors):
    #     for k in plot_constructors:
    #         if k not in _SPECIFIED_PLOTS:
    #             stream.add_plot_type(k, plot_constructors[k]())
    #         _SPECIFIED_PLOTS.add(k)

    # PLOT_DATA[name] = data
    set_plot_data_and_update(name, data, **kwargs)


def set_plot_data_and_update(name, data, **kwargs):
    PLOT_DATA[name] = data
    stream = get_dbplot_stream(**kwargs)
    stream.update()

def get_dbplot_stream(**kwargs):
    global STREAM
    if STREAM is None:
        STREAM = LiveStream(lambda: PLOT_DATA, **kwargs)
    return STREAM
