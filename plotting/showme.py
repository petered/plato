from general.nested_structures import flatten_struct
from plotting.data_conversion import vector_length_to_tile_dims
import plotting.matplotlib_backend as eplt


__author__ = 'peter'


class LiveStream(object):

    def __init__(self, callback, update_every = 1):
        self._callback = callback
        self._plots = None
        eplt.ion()  # Bad?
        self._counter = -1
        self._update_every = update_every

    def update(self):
        self._counter += 1
        if self._counter % self._update_every != 0:
            return

        struct = self._callback()
        flat_struct = flatten_struct(struct)  # list<*tuple<str, data>>

        if self._plots is None:
            self._plots = {k: eplt.get_plot_from_data(v) for k, v in flat_struct}
            n_rows, n_cols = vector_length_to_tile_dims(len(flat_struct))
            for i, (k, v) in enumerate(flat_struct):
                eplt.subplot(n_rows, n_cols, i+1)
                self._plots[k].update(v)
                eplt.title(k, fontdict = {'fontsize': 8})
            eplt.show()
        else:
            for k, v in flat_struct:
                self._plots[k].update(v)
        eplt.draw()
