import time
from plotting.live_plotting import LiveStream

__author__ = 'peter'
import numpy as np


def test_streaming():

    stream = LiveStream(lambda: {
        'images': {
            'bw_image': np.random.randn(20, 20),
            'col_image': np.random.randn(20, 20, 3),
            'vector_of_bw_images': np.random.randn(11, 20, 20),
            'vector_of_colour_images': np.random.randn(11, 20, 20, 3),
            'matrix_of_bw_images': np.random.randn(5, 6, 20, 20),
            'matrix_of_colour_images': np.random.randn(5, 6, 20, 20, 3),
            },
        'line': np.random.randn(20),
        'lines': np.random.randn(20, 3),
        'moving_point': np.random.randn(),
        'moving_points': np.random.randn(3),
        })

    for i in xrange(500):
        if i==1:
            start_time = time.time()
        elif i>1:
            print 'Average Frame Rate: %.2f FPS' % (i/(time.time()-start_time), )
        stream.update()

if __name__ == '__main__':

    test_streaming()
