from plato.interfaces.decorators import symbolic_updater, symbolic_standard
from plato.tools.optimizers import SimpleGradientDescent
from plato.tools.symbolic_graph import SymbolicGraph
import theano
from utils.graph_utils import FactorGraph
import numpy as np


class DeepBeliefNet(object):

    def __init__(self, layers, bridges):
        assert all(src in layers and dest in layers for src, dest in bridges.viewkeys()), \
            'All bridges must project to and from layers'
        self._graph = FactorGraph(variables=layers, factors=bridges)
        self._layers = layers
        self._bridges = bridges

    def propagate(self, path, signal_dict):
        path = self._graph.get_inference_path(path)
        signals = signal_dict.copy()
        for (srcs, dest), func in path.iteritems():
            signals[dest] = func(*[signals(src) for src in srcs])
        return signals

    def get_inference_function(self, input_layers, output_layers, path=None):

        @symbolic_standard
        def inference_fcn(input_signals):
            initial_signal_dict = {lay: sig for lay, sig in zip(input_layers, input_signals)}
            computed_signal_dict = self.propagate(path, signal_dict = initial_signal_dict)
            return [computed_signal_dict[lay] for lay in output_layers]

        return inference_fcn

    def get_constrastive_divergence_function(self, visible_layers, hidden_layers, up_path = [], n_gibbs = 1, persistent = False,
            optimizer = SimpleGradientDescent(eta = 0.1)):

        if len(up_path)==0:
            input_layers = visible_layers
        else:
            input_layers = up_path[0]
            assert up_path[-1] == visible_layers

        propup = self.get_inference_function(visible_layers, hidden_layers)
        free_energy = self.get_free_energy_function(visible_layers, hidden_layers)

        @symbolic_updater
        def cd_function(input_signals):

            wake_visible = input_signals if len(up_path)==0 else self.get_inference_function(input_layers, visible_layers, up_path)
            wake_hidden = propup(wake_visible)

            initial_hidden =[theano.shared(np.zeros(wh.tag.test_value.shape, dtype = theano.config.floatX), name = 'persistent_hidden_state') for wh in wake_hidden] \
                if persistent else wake_hidden

            gibbs_path = [(hidden_layers, visible_layers)] + [(visible_layers, hidden_layers), (hidden_layers, visible_layers)] * (n_gibbs-1)
            sleep_visible = self.get_inference_function(hidden_layers, visible_layers, gibbs_path)(initial_hidden)
            sleep_hidden = propup(sleep_visible)

            wake_energy = free_energy(wake_visible, wake_hidden).mean()
            sleep_energy = free_energy(sleep_visible, sleep_hidden).mean()

            all_params = sum([x.parameters for x in ([self._layers[i] for i in visible_layers]
                +[self._layers[i] for i in hidden_layers]+[self._bridges[i, j] for i in visible_layers for j in hidden_layers])], [])

            updates = optimizer(cost = wake_energy-sleep_energy, params = all_params, constants = wake_visible+sleep_visible)

            if persistent:
                updates += [(p, s) for p, s in zip(initial_hidden, sleep_hidden)]

            return updates

    def get_free_energy_function(self, visible_layers, hidden_layers):

        # TODO: Verify that computation is correct for all choices of vis/hidden layers
        # http://www.iro.umontreal.ca/~lisa/twiki/bin/view.cgi/Public/DBNEquations
        #
        #

        @symbolic_stateless
        def free_energy(visible_signals):
            pass



        return free_energy



    def get_training_function(self, visible_layer_ds, hidden_layer_ids, input_layer_ids = None,
            n_gibbs=1, persistent = False, optimizer = SimpleGradientDescent(eta = 0.01)):
        pass


# def _build_dbn_graph(layers, bridges):
#     graph_specifier = {}
#     bridge_namer = lambda src, dest: 'bridge[%s,%s]' % (src, dest)
#     for (src_layer_id, dest_layer_id), b in bridges:
#         bridge_out_id = bridge_namer(src_layer_id, dest_layer_id)
#         graph_specifier[src_layer_id, bridge_out_id] = b
#         graph_specifier[bridge_out_id, src_layer_id] = b.reverse
#     for layer_id, layer in layers:
#         graph_specifier[tuple(bridge_namer(s, d) for s, d in bridges if d == layer_id), layer_id] = layer
#     return SymbolicGraph(graph_specifier)
