from plato.interfaces.decorators import symbolic_standard
from utils.graph_utils import SignalGraph

__author__ = 'peter'


class SymbolicGraph(object):

    def __init__(self, graph_specifier):
        assert isinstance(graph_specifier, dict)
        self._graph = SignalGraph(graph_specifier)

    def get_function(self, input_signals, output_signals, path = None):

        subgraph = self._graph.get_subgraph(input_signals=input_signals, output_signals=output_signals)

        if path is None:
            path = subgraph.get_serial_order()

        @symbolic_standard
        def flow_function(inputs):
            signals = {sig_name: sig for sig_name, sig in zip(input_signals, inputs)}
            updates = []
            for function_node_identifier in path:
                src_sig_names, dest_sig_names = function_node_identifier
                node_inputs = [signals[name] for name in src_sig_names]
                node_outputs, node_updates = self._graph[function_node_identifier].symbolic_standard(node_inputs)
                signals.update({name: sig for name, sig in zip(output_signals, node_outputs)})
                updates.append(node_updates) # Note: potential problem here with shared vars - may need scan
            outputs = [signals[name] for name in output_signals]
            return outputs, updates

        return flow_function
