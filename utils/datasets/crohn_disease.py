import csv
from utils.datasets.data_splitting import split_data_by_label
from utils.datasets.datasets import DataSet
from utils.file_getter import get_file
import numpy as np

__author__ = 'peter'


def get_chrone_dataset(frac_training = 0.7):

    filename = get_file('data/is-pro.csv')

    with open(filename) as f:
        raw_data = [row for row in csv.reader(f)]

    feature_names = raw_data[2:]
    labels = [row[1] for row in raw_data[1:]]

    _, labels = np.unique(labels, return_inverse = True)

    features = np.array([[float(r) for r in row[2:]] for row in raw_data[1:]])

    x_tr, y_tr, x_ts, y_ts = split_data_by_label(features, labels, frac_training=frac_training)

    return DataSet.from_xyxy(x_tr, y_tr, x_ts, y_ts)


if __name__ == '__main__':

    from plotting.easy_plotting import ezplot, plot_data_dict
    from plotting.matplotlib_backend import ImagePlot

    ds = get_chrone_dataset()

    x_tr, y_tr, x_ts, y_ts = ds.xyxy

    plot_data_dict([
        ('Training Features', x_tr),
        ('Training Labels', y_tr[:, None]),
        ('Test Features', x_ts),
        ('Test Labels', y_ts[:, None])
        ],
        plots = {
            'Training Features': ImagePlot(),
            'Training Labels': ImagePlot(),
            'Test Features': ImagePlot(),
            'Test Labels': ImagePlot(),
            }
        )
