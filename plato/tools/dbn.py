from plato.tools.optimizers import SimpleGradientDescent
from plato.tools.symbolic_graph import SymbolicGraph


class DeepBeliefNet(object):

    def __init__(self, layers, bridges):
        assert all(src in layers and dest in layers for src, dest in bridges.viewkeys()), \
            'All bridges must project to and from layers'
        self._graph = _build_dbn_graph(layers, bridges)

    def get_inference_function(self, input_signals, output_signals, path=None):
        return self._graph.get_function(input_signals=input_signals, output_signals=output_signals, path=path)

    def get_training_function(self, visible_layer_ds, hidden_layer_ids, input_layer_ids = None,
            n_gibbs=1, persistent = False, optimizer = SimpleGradientDescent(eta = 0.01)):
        pass



def _build_dbn_graph(layers, bridges):
    graph_specifier = {}
    bridge_namer = lambda src, dest: 'bridge[%s,%s]' % (src, dest)
    for (src_layer_id, dest_layer_id), b in bridges:
        bridge_out_id = bridge_namer(src_layer_id, dest_layer_id)
        graph_specifier[src_layer_id, bridge_out_id] = b
        graph_specifier[bridge_out_id, src_layer_id] = b.reverse
    for layer_id, layer in layers:
        graph_specifier[tuple(bridge_namer(s, d) for s, d in bridges if d == layer_id), layer_id] = layer
    return SymbolicGraph(graph_specifier)
