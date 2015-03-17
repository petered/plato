from abc import ABCMeta, abstractmethod
from plotting.data_conversion import put_data_in_grid, RecordBuffer, scale_data_to_8_bit, data_to_image

__author__ = 'peter'


from matplotlib.pyplot import *


class IPlot(object):

    __metaclass__ = ABCMeta

    @abstractmethod
    def update(self):
        pass


class ImagePlot(object):

    def __init__(self, interpolation = 'nearest', show_axes = False, scale = None, aspect = 'auto', cmap = 'gray'):
        self._plot = None
        self._interpolation = interpolation
        self._show_axes = show_axes
        self._scale = scale
        self._aspect = aspect
        self._cmap = cmap

    def update(self, data):

        plottable_data = put_data_in_grid(data, clims = self._scale, cmap = self._cmap) \
            if not (data.ndim==2 or data.ndim==3 and data.shape[2]==3) else \
            data_to_image(data, clims = self._scale, cmap = self._cmap)

        if self._plot is None:
            self._plot = imshow(plottable_data, interpolation = self._interpolation, aspect = self._aspect, cmap = self._cmap)
            if not self._show_axes:
                # self._plot.axes.get_xaxis().set_visible(False)
                self._plot.axes.tick_params(labelbottom = 'off')
                self._plot.axes.get_yaxis().set_visible(False)
            # colorbar()

        else:
            self._plot.set_array(plottable_data)
        self._plot.axes.set_xlabel('%.2f - %.2f' % (np.nanmin(data), np.nanmax(data)))
            # self._plot.axes.get_caxis


class MovingImagePlot(ImagePlot):

    def __init__(self, buffer_len = 100, **kwargs):
        ImagePlot.__init__(self, **kwargs)
        self._buffer = RecordBuffer(buffer_len)

    def update(self, data):
        if np.isscalar(data):
            data = np.array([data])
        elif data.ndim != 1 and data.size == np.max(data.shape):
            data = data.flatten()
        else:
            assert data.ndim == 1

        buffer_data = self._buffer(data)
        ImagePlot.update(self, buffer_data)


class LinePlot(object):

    def __init__(self, yscale = None):
        self._plots = None
        self._yscale = yscale
        self._oldlims = (float('inf'), -float('inf'))

    def update(self, data):

        lower, upper = (np.nanmin(data), np.nanmax(data)) if self._yscale is None else self._yscale

        if self._plots is None:
            self._plots = plot(np.arange(-data.shape[0]+1, 1), data)
            for p, d in zip(self._plots, data[None] if data.ndim==1 else data.T):
                p.axes.set_xbound(-len(d), 0)
                if lower != upper:  # This happens in moving point plots when there's only one point.
                    p.axes.set_ybound(lower, upper)
        else:
            for p, d in zip(self._plots, data[None] if data.ndim==1 else data.T):
                p.set_ydata(d)
                if lower!=self._oldlims[0] or upper!=self._oldlims[1]:
                    p.axes.set_ybound(lower, upper)

        self._oldlims = lower, upper


class MovingPointPlot(LinePlot):

    def __init__(self, buffer_len=100, **kwargs):
        LinePlot.__init__(self, **kwargs)
        self._buffer = RecordBuffer(buffer_len)

    def update(self, data):
        if not np.isscalar(data):
            data = data.flatten()

        buffer_data = self._buffer(data)
        LinePlot.update(self, buffer_data)


class TextPlot(IPlot):

    def __init__(self, max_history = 8):
        self._buffer = RecordBuffer(buffer_len = max_history, initial_value='')
        self._max_history = 10
        self._text_plot = None

    def update(self, string):
        if not isinstance(string, basestring):
            string = str(string)
        history = self._buffer(string)
        full_text = '\n'.join(history)
        if self._text_plot is None:
            ax = gca()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            self._text_plot = ax.text(0.05, 0.05, full_text)
        else:
            self._text_plot.set_text(full_text)


def get_plot_from_data(data, mode, **plot_preference_kwargs):

    assert mode in ('live', 'static')

    if mode == 'live':
        plot = get_live_plot_from_data(data, **plot_preference_kwargs)
    else:
        plot = get_static_plot_from_data(data, **plot_preference_kwargs)
    return plot


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
