from plato.tools.pretrained_networks.vggnet import get_vgg_net, get_vggnet_labels, im2vgginput
from utils.datasets.art_gallery import get_image
import numpy as np
import pytest
__author__ = 'peter'


@pytest.mark.slowtest
def test_vggnet():

    vggnet = get_vgg_net()
    f_predict = vggnet.compile(add_test_values = False)
    inp = im2vgginput(get_image('heron'))
    out = f_predict(inp)
    label = get_vggnet_labels()[np.argmax(out)]
    assert label == 'American egret, great white heron, Egretta albus'


@pytest.mark.slowtest
def test_norm_vggnet():

    vggnet = get_vgg_net(normalized_version=True)
    f_predict = vggnet.compile(add_test_values = False)
    inp = im2vgginput(get_image('heron'))
    out = f_predict(inp)
    label = get_vggnet_labels()[np.argmax(out)]
    assert label == 'American egret, great white heron, Egretta albus'


if __name__ == '__main__':
    # test_vggnet()
    test_norm_vggnet()
