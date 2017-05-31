from functools import partial

import theano
from artemis.experiments.experiment_record import Experiment
from artemis.experiments.deprecated import ExperimentLibrary
from artemis.general.ezprofile import EZProfiler
from artemis.ml.datasets.art_gallery import get_image
from plato.core import symbolic
from plato.tools.pretrained_networks.vggnet import im2vgginput, get_vgg_net


def profile_vggnet(force_shared_parameters = True, add_test_values = True):

    img = get_image('dinnertime', size = (224, 224))

    img_data = im2vgginput(img)

    print theano.config.floatX

    @symbolic
    def get_vgg_features(data):
        vggnet = get_vgg_net(force_shared_parameters=force_shared_parameters)
        prof.lap('Loaded VGGNet')
        out = vggnet(data)
        prof.lap('Done Symbolic Pass')
        return out

    func = get_vgg_features.compile(add_test_values=add_test_values)

    with EZProfiler(record_stop=False) as prof:
        out = func(img_data)
        prof.lap('Compiled')
        out = func(img_data)
        prof.lap('Second Pass')


ExperimentLibrary.profile_vggnet = Experiment(
    description='Profile ',
    function=partial(profile_vggnet),
    versions=dict(
        baseline = dict(force_shared_parameters = True, add_test_values = True),
        no_test_values = dict(force_shared_parameters = True, add_test_values = False),
        no_shared = dict(force_shared_parameters = False, add_test_values = True),
        no_shared_or_test_values = dict(force_shared_parameters = False, add_test_values = False),
    ),
    current_version='no_shared_or_test_values',
    conclusion = """

    baseline: On Standard options with GPU: GeForce GTX TITAN X (CNMeM is disabled, CuDNN not available)
          Loaded VGGNet: Elapsed time is 3.849s
          Done Symbolic Pass: Elapsed time is 195.7s
          Compiled: Elapsed time is 1.668s
          Second Pass: Elapsed time is 0.02801s
          Total: 201.2s

    no_test_values:
          Loaded VGGNet: Elapsed time is 3.781s
          Done Symbolic Pass: Elapsed time is 0.5406s
          Compiled: Elapsed time is 1.985s
          Second Pass: Elapsed time is 0.02072s
          Total: 6.327s

    no_shared:
          Loaded VGGNet: Elapsed time is 3.797s
          Done Symbolic Pass: Elapsed time is 190.1s
          Compiled: Elapsed time is 8.801s
          Second Pass: Elapsed time is 0.02044s
          Total: 202.7s

    no_shared_or_test_values
          Loaded VGGNet: Elapsed time is 3.997s
          Done Symbolic Pass: Elapsed time is 0.2553s
          Compiled: Elapsed time is 6.051s
          Second Pass: Elapsed time is 0.02029s
          Total: 10.32s

    no_test_values: CPU: floatX = float32
          Loaded VGGNet: Elapsed time is 3.717s
          Done Symbolic Pass: Elapsed time is 1.553s
          Compiled: Elapsed time is 7.181s
          Second Pass: Elapsed time is 0.6635s
          Total: 13.11s

    Conclusion... the problem is test values.  The first pass, with test values, takes an eternity.
    """
    )


if __name__ == '__main__':
    ExperimentLibrary.profile_vggnet.run()
