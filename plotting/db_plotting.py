from plotting.live_plotting import LiveStream

__author__ = 'peter'


PLOT_DATA = {}

STREAM = None


def dbplot(data, name = None, plot_mode = 'static', **kwargs):
    """
    Quick plot of some variable - you can call this in a loop and it will know to update the
    same plot.
    :param data: The data to plot
    :param name: The name of this plot (you need to specify this if you want to make more than one plot)
    :param plot_mode: Affects the type of plots generated.
        'live' is more appropriate when you're monitoring something and want an online plot with memory
        'static' is better for step-by-step debugging, or plotting from the debugger.
    """

    global STREAM
    if STREAM is None:
        STREAM = LiveStream(lambda: PLOT_DATA, plot_mode=plot_mode, **kwargs)
    if not isinstance(name, str):
        name = str(name)
    PLOT_DATA[name] = data
    STREAM.update()
