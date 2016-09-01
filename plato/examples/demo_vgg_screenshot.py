from artemis.fileman.smart_io import smart_load
from artemis.general.should_be_builtins import bad_value
from artemis.plotting.db_plotting import dbplot
from plato.tools.pretrained_networks.vggnet import get_vgg_net, im2vgginput, get_vgg_label_at
import numpy as np
import time
__author__ = 'peter'
import os

"""
This program scans for photos from the webcam, then processes them with vggnet.

It looks for photos in the directory that the "Photo Booth" application on MacOS puts photos from the webcam.

The first time processing the image should take ~20s, each time after that ~1s.

To Use (only works on Mac)
- Open PhotoBooth.
- Take a screenshot
- A window should pop up showing the image with the label that the network decides on
- Repeat
"""


def get_photo_dir():
    return os.path.join(os.path.expanduser('~'), 'Pictures/Photo Booth Library/Pictures')


def get_all_photos():
    return os.listdir(get_photo_dir())


def get_latest_screenshot():
    photodir = get_photo_dir()
    files = os.listdir(photodir)
    latest = sorted(files)[-1]
    full_path = os.path.join(photodir, latest)
    return full_path


def classify(f, im_path):
    im = smart_load(im_path)
    print 'Processing image... "%s"' % (im_path, )
    inputs = im2vgginput(im)
    out = f(inputs)
    amax = np.argmax(out[0])
    label = get_vgg_label_at(amax)
    print 'Done.'
    dbplot(np.rollaxis(inputs[0], 0, 3)[..., ::-1], 'Photo', title="{label}: {pct}%".format(label = label, pct = out[0, amax, 0, 0]*100))


def demo_photobooth():
    old_photos = set(get_all_photos())
    f = get_vgg_net().compile(add_test_values = False)
    print 'Take a screenshot with PhotoBooth'
    while True:
        new_photos = set(get_all_photos()).difference(old_photos)
        if len(new_photos) != 0:
            classify(f, os.path.join(get_photo_dir(), new_photos.pop()))
            old_photos = set(get_all_photos())
        time.sleep(.1)


def demo_file_path():

    f = get_vgg_net().compile(add_test_values = False)
    while True:
        im_path = raw_input("Enter Image Path: ")
        classify(f, im_path)


if __name__ == '__main__':

    VERSION = "photobooth"

    if VERSION == 'photobooth':
        demo_photobooth()
    elif VERSION == 'file':
        demo_file_path()
    else:
        bad_value(VERSION)
