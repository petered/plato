from collections import OrderedDict, namedtuple

import theano
from plato.tools.convnet.convnet import ConvLayer, Nonlinearity, Pooler, ConvNet, ConvolverSpec, NonlinearitySpec, \
    PoolerSpec
from fileman.file_getter import get_file
from general.should_be_builtins import bad_value, memoize
from scipy.io import loadmat
import numpy as np
from theano.gof.graph import Variable
import theano.tensor as tt

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
    More Details: http://cs231n.github.io/convolutional-networks/#case

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
            # layer = ConvolverSpec(w=w, b=b, mode = 'full' if layer_name.startswith('fc') else 'same' if layer_name.startswith('conv') else bad_value(layer_name));
            layer = ConvolverSpec(w=w, b=b, mode = 'full' if layer_name.startswith('fc') else 'same' if layer_name.startswith('conv') else bad_value(layer_name))
        elif layer_type in ('relu', 'softmax'):
            # layer = NonlinearitySpec(layer_type)
            layer = NonlinearitySpec(layer_type)
        elif layer_type == 'pool':
            # layer = PoolerSpec(
            #     region = tuple(struct[3][0].astype(int)),
            #     stride = tuple(struct[4][0].astype(int)),
            #     mode=struct[2][0])
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







def get_vgg_net(up_to_layer=None, force_shared_parameters=True, scale_biases = 1):
    """
    Load the 19-layer VGGNet.
    Info: https://gist.github.com/ksimonyan/3785162f95cd2d5fee77#file-readme-md
    More Details: http://cs231n.github.io/convolutional-networks/#case

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
    layer_specs = get_vgg_layer_specifiers(up_to_layer=up_to_layer)

    return ConvNet.from_init(layer_specs, input_shape=(3, 224, 224))

    # layer_constructor = lambda spec: {
    #     ConvolverSpec: lambda: ConvLayer(
    #         w=spec.w,
    #         b=spec.b if scale_biases==1 else spec.b*scale_biases,
    #         force_shared_parameters=force_shared_parameters,
    #         border_mode= {'full': 0, 'same': 1}[spec.mode],
    #         filter_flip=False
    #         ),
    #     NonlinearitySpec: Nonlinearity(spec.type),
    #     PoolerSpec: lambda: Pooler(region=spec.region, stride=spec.stride, mode=spec.mode)
    #     }[spec.__class__]()
    # layers = OrderedDict((name, layer_constructor(spec)) for name, spec in layer_specs.iteritems())
    # return ConvNet(layers)


    # filename = get_file(
    #     relative_name='data/vgg-19.mat',
    #     url='http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat',
    # )
    # network_params = loadmat(filename)
    #
    # def struct_to_layer(struct):
    #     layer_type = struct[1][0]
    #     layer_name = str(struct[0][0])
    #     assert isinstance(layer_type, basestring)
    #     if layer_type == 'conv':
    #         w_orig = struct[2][0, 0]  # (n_rows, n_cols, n_in_maps, n_out_maps)
    #         # (n_out_maps, n_in_maps, n_rows, n_cols)  (Theano conventions)
    #         w = w_orig.T.swapaxes(2, 3)
    #         b = struct[2][0, 1][:, 0]
    #         padding = 0 if layer_name.startswith('fc') else 1 if layer_name.startswith('conv') else bad_value(layer_name)
    #         layer = ConvLayer(w, b, force_shared_parameters=force_shared_parameters, border_mode=padding, filter_flip=False)  # Note: Should filter_flip be true...? Need to check
    #     elif layer_type in ('relu', ):
    #         layer = Nonlinearity(layer_type)
    #     elif layer_type == 'softmax':
    #         layer = Nonlinearity(softmax)
    #     elif layer_type == 'pool':
    #         layer = Pooler(region=tuple(struct[3][0].astype(int)), stride=tuple(
    #             struct[4][0].astype(int)), mode=pooling_mode)
    #     else:
    #         raise Exception(
    #             "Don't know about this '%s' layer type." % layer_type)
    #     return layer_name, layer
    #
    # print 'Loading VGG Net...'
    # network_layers = OrderedDict(struct_to_layer(network_params['layers'][0, i][
    #                              0, 0]) for i in xrange(network_params['layers'].shape[1]))
    #
    # if up_to_layer is not None:
    #     if isinstance(up_to_layer, (list, tuple)):
    #         up_to_layer = network_layers.keys()[max(
    #             network_layers.keys().index(layer_name) for layer_name in up_to_layer)]
    #     layer_names = [network_params['layers'][0, i][0, 0][0][0]
    #                    for i in xrange(network_params['layers'].shape[1])]
    #     network_layers = OrderedDict((k, network_layers[k]) for k in layer_names[
    #                                  :layer_names.index(up_to_layer) + 1])
    #     print 'Done.'
    # return ConvNet(network_layers)




def im2vgginput(im):
    """
    :param im: A (size_y, size_x, 3) array representing a RGB image on a [0, 255] scale, or a
        (n_samples, size_y, size_x, 3) array representing an array of such images.
    :returns: A (1, 3, size_y, size_x) array representing the BGR image that's ready to feed into VGGNet

    """
    centered_bgr_im = im[..., ::-1] - np.array([103.939, 116.779, 123.68])
    # if im.ndim==4:
    #     feature_map_im = centered_bgr_im.dimshuffle('x', 2, 0, 1) if isinstance(centered_bgr_im, Variable) else np.rollaxis(centered_bgr_im, 2, 0)[None, :, :, :]
    # elif im.ndim==3:
    feature_map_im = centered_bgr_im.dimshuffle('x', 2, 0, 1) if isinstance(centered_bgr_im, Variable) else np.rollaxis(centered_bgr_im, 2, 0)[None, :, :, :]
    # else:
    #     raise Exception('bah')
    return feature_map_im.astype(theano.config.floatX)


def vgginput2im(feat):
    """
    :param feat: A (1, 3, size_y, size_x) array representing the BGR image that's ready to feed into VGGNet
    :returns: A (size_y, size_x, 3) array representing a RGB image.
    """
    bgr_im = (feat.dimshuffle(0, 2, 3, 1) if isinstance(feat, Variable) else np.rollaxis(feat, 0, 2))[0, :, :, :]
    decentered_rgb_im = (bgr_im + np.array([103.939, 116.779, 123.68]))[:, :, ::-1]
    return decentered_rgb_im


def get_named_labels():
    file_loc = get_file(
        relative_name='data/labels.txt',
        url = 'https://raw.githubusercontent.com/HoldenCaulfieldRye/caffe/master/data/ilsvrc12/synset_words.txt')
    with open(file_loc) as f:
        lines = f.readlines()
    labels = [line[10:-1] for line in lines]
    return labels
