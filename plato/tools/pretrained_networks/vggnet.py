from collections import OrderedDict
import pickle
from artemis.fileman.disk_memoize import memoize_to_disk
from numpy.testing.utils import assert_raises
from plato.tools.convnet.conv_specifiers import ConvolverSpec, NonlinearitySpec, PoolerSpec
import theano
from plato.tools.convnet.convnet import ConvNet
from artemis.fileman.file_getter import get_file
from artemis.general.should_be_builtins import bad_value, memoize
from scipy.io import loadmat
import numpy as np
from theano.gof.graph import Variable


__author__ = 'peter'


type_matches = lambda collection, klass: np.array(
    [isinstance(x, type) for x in list], dtype=np.bool)

find_nth_match = lambda bool_arr, n: np.nonzeros


@memoize
def get_vgg_layer_specifiers(up_to_layer=None):
    """
    Load the 19-layer VGGNet from the mat file and produce a list of layer specifications which can be used to create
    layers in your architecture of choice.
    Info: https://gist.github.com/ksimonyan/3785162f95cd2d5fee77#file-readme-md
    More Details: http://cs231n.github.io/convolutional-convnets/#case

    :param up_to_layer: The layer to stop at.  Or a list of layers, in which case the network will go to the highest.
        Layers are identified by their string names:
        ['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3', 'conv4_1',
        'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4', 'conv5_1', 'relu5_1',
        'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5', 'fc6', 'relu6', 'fc7', 'relu7',
        'fc8', 'prob']
    :param force_shared_parameters: Create net with shared paremeters.
    :return: An OrderedDict<str,PrimativeSpecifier> where PrimativeSpecifier objects represent the layers of the network.
    """

    filename = get_file(
        relative_name='data/vgg-19.mat',
        url='http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat',
    )
    network_params = loadmat(filename)

    def struct_to_layer(struct):
        layer_type = struct[1][0]
        layer_name = str(struct[0][0])
        assert isinstance(layer_type, basestring)
        if layer_type == 'conv':
            w_orig = struct[2][0, 0]  # (n_rows, n_cols, n_in_maps, n_out_maps)
            w = w_orig.T.swapaxes(2, 3)
            b = struct[2][0, 1][:, 0]
            layer = ConvolverSpec(w=w, b=b, mode = 'valid' if layer_name.startswith('fc') else 'same' if layer_name.startswith('conv') else bad_value(layer_name))
        elif layer_type in ('relu', 'softmax'):
            layer = NonlinearitySpec(layer_type)
        elif layer_type == 'pool':
            layer = PoolerSpec(
                region = tuple(struct[3][0].astype(int)),
                stride = tuple(struct[4][0].astype(int)),
                mode=struct[2][0])
        else:
            raise Exception(
                "Don't know about this '%s' layer type." % layer_type)
        return layer_name, layer

    print 'Loading VGG Net...'
    network_layers = OrderedDict(struct_to_layer(network_params['layers'][0, i][
                                 0, 0]) for i in xrange(network_params['layers'].shape[1]))

    if up_to_layer is not None:
        if isinstance(up_to_layer, (list, tuple)):
            up_to_layer = network_layers.keys()[max(
                network_layers.keys().index(layer_name) for layer_name in up_to_layer)]
        layer_names = [network_params['layers'][0, i][0, 0][0][0]
                       for i in xrange(network_params['layers'].shape[1])]
        network_layers = OrderedDict((k, network_layers[k]) for k in layer_names[
                                     :layer_names.index(up_to_layer) + 1])
    print 'Done.'
    return network_layers


def get_vgg_net(up_to_layer=None, force_shared_parameters=True, scale_biases = 1, normalized=False):
    """
    Load the 19-layer VGGNet.
    Info: https://gist.github.com/ksimonyan/3785162f95cd2d5fee77#file-readme-md
    More Details: http://cs231n.github.io/convolutional-convnets/#case

    :param up_to_layer: The layer to stop at.  Or a list of layers, in which case the network will go to the highest.
        Layers are identified by their string names:
        ['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3', 'conv4_1',
        'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4', 'conv5_1', 'relu5_1',
        'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5', 'fc6', 'relu6', 'fc7', 'relu7',
        'fc8', 'prob']
    :param force_shared_parameters: Create net with shared paremeters.
    :return: A ConvNet object representing the VGG network.
    """
    if normalized:
        layer_specs = get_normalized_vgg_net(up_to_layer=up_to_layer)
    else:
        layer_specs = get_vgg_layer_specifiers(up_to_layer=up_to_layer)

    if scale_biases != 1:
        for spec in layer_specs.values():
            if isinstance(spec, ConvolverSpec):
                spec.b *= scale_biases

    return ConvNet.from_init(layer_specs, input_shape=(3, 224, 224), force_shared_parameters = force_shared_parameters)


def get_normalized_vgg_net(up_to_layer=None, force_shared_parameters=True):
    """
    Load the normalized version of VGG19 discussed here: https://bethgelab.org/deepneuralart/

    """

    norm_vgg19_file = get_file(
        relative_name='data/norm-vgg-19.pkl',
        url = 'https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/vgg19_normalized.pkl',
    )
    with open(norm_vgg19_file) as f:
        vgg_struct = pickle.load(f)

    layer_names = ['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3', 'conv4_1',
        'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4', 'conv5_1', 'relu5_1',
        'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5', 'fc6', 'relu6', 'fc7', 'relu7',
        'fc8', 'prob']

    if isinstance(up_to_layer, list):
        up_to_layer = up_to_layer[np.argmax([layer_names.index(layer_name) for layer_name in up_to_layer])]

    assert up_to_layer is not None and layer_names.index(up_to_layer) < layer_names.index('fc6'), "This can only be used to load the convolutional portion of vggnet.  Set "

    net_spec = OrderedDict()
    param_iterator = (p for p in vgg_struct['param values'])
    for layer_name in layer_names:
        if layer_name.startswith('conv'):
            w = param_iterator.next()
            b = param_iterator.next()
            assert w.ndim==4 and b.ndim==1
            layer = ConvolverSpec(w=w, b=b, mode = 'same')
        elif layer_name.startswith('relu'):
            layer = NonlinearitySpec('relu')
        elif layer_name.startswith('pool'):
            layer = PoolerSpec(region=2, stride=2, mode='max')
        elif layer_name.startswith('fc'):
            w = param_iterator.next()
            b = param_iterator.next()
            # Here we'll express the "full" layers as convolutional.
            if layer_name == 'fc6':
                w = w.T.reshape(4096, 512, 7, 7)
            elif layer_name == 'fc7':
                w = w.T.reshape(4096, 4096, 1, 1)
            elif layer_name == 'fc8':
                w = w.T.reshape(1000, 4096, 1, 1)
            else:
                bad_value(layer_name)
            layer = ConvolverSpec(w=w, b=b, mode = 'valid')
        elif layer_name == 'prob':
            layer = NonlinearitySpec('softmax')
        else:
            raise Exception("Don't know how to handle layer: '%s'" % (layer_name, ))
        net_spec[layer_name] = layer
        if layer_name == up_to_layer:
            break

    if up_to_layer is None:
        assert_raises(StopIteration)
    return net_spec


def im2vgginput(im, shaping_mode = 'squeeze', already_bgr = False):
    """
    :param im: A (size_y, size_x, 3) array representing a RGB image on a [0, 255] scale, or a
        (n_samples, size_y, size_x, 3) array representing an array of such images.
    :param shaping_mode: 'squeeze': Squeezes the image into the desired shape.
        'crop': Crops the center region (of the desired shape) out.
    :returns: A (n_samples, 3, 224, 224) array representing the BGR image that's ready to feed into VGGNet

    """
    if not isinstance(im, np.ndarray) or im.ndim==4:
        return np.concatenate([im2vgginput(m, shaping_mode = shaping_mode) for m in im]) if len(im)>0 else np.zeros((0, 3, 224, 224))

    if im.ndim==2:
        im = np.repeat(im[:, :, None], repeats=3, axis=2)

    if any(m.shape[-2:-1] != (224, 224) for m in im):
        if shaping_mode == 'squeeze':
            from scipy.misc.pilutil import imresize
            # TODO: Test!
            im = imresize(im, size=(224, 224))
        elif shaping_mode == 'crop':
            current_shape = im.shape[-3:-1]
            assert current_shape[0]>=224 and current_shape[1]>=224, "Don't currently have padding implemented"
            row_start, col_start = [(c-224)/2 for c in current_shape]
            im = im[..., row_start:row_start+224, col_start:col_start+224, :]
        else:
            raise Exception('Unknown shaping mode: "%s"' % (shaping_mode, ))

    bgr_im = im if already_bgr else im[..., ::-1]
    centered_bgr_im = bgr_im - np.array([103.939, 116.779, 123.68])
    feature_map_im = np.rollaxis(centered_bgr_im, -1, -3)
    if feature_map_im.ndim==3:
        feature_map_im = feature_map_im[None, ...]
    return feature_map_im.astype(theano.config.floatX)


def vgginput2im(feat):
    """
    :param feat: A (1, 3, size_y, size_x) array representing the BGR image that's ready to feed into VGGNet
    :returns: A (size_y, size_x, 3) array representing a RGB image.
    """
    bgr_im = (feat.dimshuffle(0, 2, 3, 1) if isinstance(feat, Variable) else np.rollaxis(feat, 0, 2))[0, :, :, :]
    decentered_rgb_im = (bgr_im + np.array([103.939, 116.779, 123.68]))[:, :, ::-1]
    return decentered_rgb_im


def get_vggnet_labels():
    file_loc = get_file(
        relative_name='data/labels.txt',
        url = 'https://raw.githubusercontent.com/HoldenCaulfieldRye/caffe/master/data/ilsvrc12/synset_words.txt')
    with open(file_loc) as f:
        lines = f.readlines()
    labels = [line[10:-1] for line in lines]
    return labels


_VGG_LABELS = None


def get_vgg_label_at(label_index, short=False):
    global _VGG_LABELS
    if _VGG_LABELS is None:
        _VGG_LABELS = get_vggnet_labels()
    label = _VGG_LABELS[label_index]
    if short:
        label = label[:label.index(',')] if ',' in label else label
    return label
