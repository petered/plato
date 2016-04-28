from collections import OrderedDict
import numpy as np
from abc import ABCMeta, abstractmethod
# from general.should_be_builtins import bad_value

from bokeh.document import Document
from bokeh.models import Plot
from bokeh.client import push_session
# from bokeh.models import Plot
from bokeh.plotting import Figure, figure
# from bokeh.charts import Area
from bokeh.palettes import Spectral6

import matplotlib

_SESSION_COUNTER = 0
_SESSIONS = OrderedDict()
_CURRENT_SESSION = None
_CURRENT_PLOT = None
_URL = "default"



def set_url(url):
    global _URL
    _URL = url


def get_name():
    global _SESSION_COUNTER
    _SESSION_COUNTER += 1
    return "Figure_" + str(_SESSION_COUNTER)


class FakeFuckingFigure():
    def __init__(self, session):
        self.session = session

    def clf(self):
        _CURRENT_SESSION.document.clear()
        global _CURRENT_PLOT
        _CURRENT_PLOT = None



def figure(*args,**kwargs):
    return FakeFuckingFigure(_session(*args,**kwargs))

def _session(name=None):
    if name == None:
        name = get_name()
    doc = Document()
    session = push_session(document=doc, session_id=name, url=_URL) #TODO FILL IN Details
    session.show()
    global _CURRENT_SESSION
    _CURRENT_SESSION = session
    return session

def _get_or_make_session():
    return figure() if _CURRENT_SESSION is None else _CURRENT_SESSION

def _plot(model = None, **kwargs):
    print(kwargs)
    plot = model(**kwargs)
    global _CURRENT_SESSION
    _CURRENT_SESSION.document.add_root(plot)
    global _CURRENT_PLOT
    _CURRENT_PLOT = plot
    return plot

def _get_or_make_plot(model, **kwargs):
        return _plot(model, **kwargs) if _CURRENT_PLOT is None else _CURRENT_PLOT

def make_plot(model, **kwargs):
    return _plot(model, **kwargs)

def subplot(rows, cols, num, **kwargs):
    # Set this up as gridplot
    # if num == 0:
        # create new gridplot
    # else:
        # add to existing gridplot

    session = _get_or_make_session()
    ax = make_plot(Figure, **kwargs)
    global _CURRENT_PLOT
    _CURRENT_PLOT = ax
    return ax

def draw():
    pass

def show():
    pass

def isinteractive():
    pass

def interactive(bool):
    pass

def gca():
    return _CURRENT_PLOT

def title(s, *args, **kwargs):
    gca().title = s

def plot(*args, **kwargs):

    session = _get_or_make_session()
    figure = _get_or_make_plot(Figure, **kwargs)

    if isinstance(args[0], np.ndarray) and isinstance(args[1], np.ndarray):
        x_data, y_data = args[:2]
    else:
        x_data = np.arange(len(args[0]))
        y_data = args[0]


    if x_data.ndim == 1:
        x_data = np.expand_dims(x_data,0)

    if y_data.shape != x_data.shape:
        if y_data.T.shape == x_data.shape:
            x_data = x_data.T
        elif y_data.shape[0] % x_data.shape[0] == 0:
                x_data = np.repeat(x_data,y_data.shape[0],0)
        else:
            print ("x-axis data and y-axis data not correct for multi-line plot")

    return figure.multi_line(xs = x_data.tolist(), ys = y_data.tolist(), color = Spectral6[:y_data.shape[0]], line_width = 2 )





class LinePlot(object):

    def __init__(self, yscale = None, **kwargs):
        self._plots = None
        self._yscale = yscale
        self._oldlims = (float('inf'), -float('inf'))
        self.kw = kwargs

    def update(self, data):
        if self._plots is None:
            # import pdb
            # pdb.set_trace()
            self._plots = plot(np.arange(-data.shape[0]+1, 1), data.T, **self.kw)
        else:
            self._plots.data_source.data["ys"] = data.T



class ImagePlot(object):

    def __init__(self, interpolation = 'nearest', show_axes = False, clims = None, aspect = 'auto', cmap = 'gray'):
        self._plot = None
        self._interpolation = interpolation
        self._show_axes = show_axes
        self._clims = clims
        self._aspect = aspect
        self._cmap = cmap

    def update(self, data):

        if data.ndim == 1:
            data = data[None]

        clims = ((np.nanmin(data), np.nanmax(data)) if data.size != 0 else (0, 1)) if self._clims is None else self._clims

        plottable_data = put_data_in_grid(data, clims = clims, cmap = self._cmap) \
            if not (data.ndim == 2 or data.ndim == 3 and data.shape[2] == 3) else \
            data_to_image(data, clims = clims, cmap = self._cmap)

        if self._plot is None:
            self._plot = imshow(plottable_data, interpolation = self._interpolation, aspect = self._aspect, cmap = self._cmap)
            if not self._show_axes:
                # self._plot.axes.get_xaxis().set_visible(False)
                self._plot.axes.tick_params(labelbottom = 'off')
                self._plot.axes.get_yaxis().set_visible(False)
            # colorbar()

        else:
            self._plot.set_array(plottable_data)
        self._plot.axes.set_xlabel('%.2f - %.2f' % clims)
            # self._plot.axes.get_caxis


if __name__ == "__main__":
    data = np.array([
  [ 0.33282588, -0.17099474],
 [ 0.33028431, -1.51028199],
 [-0.09953188,  2.24461989],
 [-0.31208577, -0.20377033],
 [-1.43971886,  0.03359312],
 [ 0.76929195, -0.08524877],
 [ 0.91306424, -0.85165996],
 [ 0.87956879,  1.41940303],
 [-0.04087287, -0.65596172],
 [-1.62511259,  0.27262459]])
    set_url("http://146.50.149.168:5006")
    LP = LinePlot()
    LP.update(data)
    print("Done")



def get_plot_from_data(data, mode, **plot_preference_kwargs):

    return \
        get_live_plot_from_data(data, **plot_preference_kwargs) if mode == 'live' else \
        get_static_plot_from_data(data, **plot_preference_kwargs) if mode == 'static' else \
        ImagePlot(**plot_preference_kwargs) if mode == 'image' else \
        bad_value(mode, 'Unknown plot modee: %s' % (mode, ))


def get_live_plot_from_data(data, line_to_image_threshold = 8, cmap = 'gray'):

    if isinstance(data, basestring):
        return TextPlot()

    is_scalar = np.isscalar(data) or data.shape == ()
    if is_scalar:
        data = np.array(data)

    is_1d = not is_scalar and data.size == np.max(data.shape)
    few_values = data.size < line_to_image_threshold

    if is_scalar or is_1d and few_values:
        return MovingPointPlot()
    elif is_1d:
        return MovingImagePlot()
    elif data.ndim == 2 and data.shape[1]<line_to_image_threshold:
        return LinePlot()
    elif data.ndim in (2, 3, 4, 5):
        return ImagePlot(cmap=cmap)
    else:
        raise NotImplementedError('We have no way to plot data of shape %s.  Make one!' % (data.shape, ))


def get_static_plot_from_data(data, line_to_image_threshold=8, cmap = 'gray'):

    if isinstance(data, basestring):
        return TextPlot()

    is_scalar = np.isscalar(data) or data.shape == ()
    if is_scalar or data.size==1:
        return TextPlot()

    is_1d = not is_scalar and data.size == np.max(data.shape)
    if is_1d:
        n_unique = len(np.unique(data))
        if n_unique == 2:
            return ImagePlot(cmap=cmap)
        else:
            return LinePlot()
    elif data.ndim == 2 and data.shape[1] < line_to_image_threshold:
        return LinePlot()
    else:
        return ImagePlot(cmap=cmap)
