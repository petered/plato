__author__ = 'peter'

"""
Here we define functions for dealing with a special type of graph, which we call a "SignalGraph".

Note: We may in the future redo this in more general terms with networkx, but at present it looks like
networkx does not really provide anything and requires a new dependency.
"""


class SignalGraph(set):
    """
    A SignalGraph is a bipartite directed graph, where the two types of nodes are ("signals" and "functions").
    Signals are given explicit names, and functions are referenced by the signals they consume/produce.  Each
    signal may be produced by only one function.
    """

    def __init__(self, graph_specifier):
        assert isinstance(graph_specifier, set) and all(len(el)==2 for el in graph_specifier), \
            'Your graph specifier must be a set of 2-tuples'
        clean_graph = {(_tuplefy_singles(se), _tuplefy_singles(de)) for se, de in graph_specifier.iteritems()}
        for se, de in clean_graph:
            self.add((se, de))

    def get_predecessor_signals(self, signal, memo = None):

        if memo is None:
            memo = {}
        elif signal in memo:
            return memo[signal]
        immediate_predecessors, _ = self.get_function_producing_signal(signal)
        all_predecessors = {}
        for pred_signal in immediate_predecessors:
            sub_predecessors = self.get_predecessor_signals(pred_signal, memo=memo)
            all_predecessors = all_predecessors.union(sub_predecessors)

    def get_successor_signals(self, signal):
        return self.reverse().get_predecessor_signals(signal)

    def get_predecessor_graph(self, signal):
        """ Return the graph of all functions leading to the signal """
        predecessor_signals = self.get_predecessor_signals(signal)
        return self.filter_graph(lambda src_signals, dest_signals: any(d in predecessor_signals for d in dest_signals))

    def get_successor_graph(self, signal):
        """ Return the graph of all functions downstream from the signal """
        return self.reverse().get_predecessor_graph(signal)

    def filter_graph(self, rule):
        """
        Return a new graph obtained by filtering the current graph by the given rule.
        :param rule: A function taking (src_signals, dest_signals) and returning a boolean
        :return: A SignalGraph which is a subset of the current graph.
        """
        return SignalGraph({(s, d) for s, d in self if rule(s, d)})

    def get_function_producing_signal(self, signal):
        functions_producing_signal = [(s, d) for s, d in self if signal in d]
        if len(functions_producing_signal) == 0:
            return None
        else:
            assert len(functions_producing_signal) == 1, 'Invalid Graph: You cannot have 2 functions producing the same signal'
            return functions_producing_signal[0]

    def get_functions_consuming_signal(self, signal):
        return {(s, d) for s, d in self if signal in s}

    def get_all_signals(self):
        return set.union((self.union(s, d) for s, d in self))

    def get_input_signals(self):
        return {s for s in self.get_all_signals() if self.get_function_producing_signal(s) is None}

    def get_output_signals(self):
        return {s for s in self.get_all_signals() if len(self.get_functions_consuming_signal(s))==0}

    def get_subgraph(self, input_signals = None, output_signals = None):

        if input_signals is None:
            input_signals = self.get_input_signals()

        if output_signals is None:
            output_signals = self.get_output_signals()

        graph_of_required_functions = self.get_successor_graph(output_signals)
        graph_of_available_functions = self.get_successor_graph(input_signals)
        subgraph = SignalGraph.intersection(graph_of_required_functions, graph_of_available_functions)

        assert subgraph.get_input_signals() == input_signals, \
            'The provided input signals %s are not enough to compute the subgraph %s'

        return subgraph


def _tuplefy_singles(edge_specifier):
    return edge_specifier if isinstance(edge_specifier, tuple) else edge_specifier