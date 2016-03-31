from fileman.experiment_record import ExperimentLibrary, Experiment
from fileman.file_getter import get_file
from general.ezprofile import EZProfiler
from liquid_style.pretrained_networks import get_vgg_net
from plato.core import symbolic
from plato.tools.pretrained_networks.vggnet import im2vgginput
from utils.datasets.art_gallery import get_image


def profile_vggnet():

    img = get_image('dinnertime', size = (224, 224))

    img_data = im2vgginput(img)

    @symbolic
    def forward_pass(data, net):
        out = net(data)
        prof.lap('Done pass')


    with EZProfiler() as prof:

        vggnet = get_vgg_net()

        prof.lap('Loaded VGGNet')

        func = forward_pass.compile(fixed_args = dict(net=vggnet))

        prof.lap('Defined Func')

        out = func(img_data)

        prof.lap('First-pass and compile')

        out = func(img_data)

        prof.lap('Compiled Pass')


ExperimentLibrary.profile_vggnet = Experiment(
    description='Profile ',
    function=profile_vggnet,
    )


if __name__ == '__main__':
    ExperimentLibrary.profile_vggnet.run()
