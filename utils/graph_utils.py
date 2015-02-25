__author__ = 'peter'

"""
Here we define functions for dealing with a special type of graph, which we call a "SignalGraph".

Note: We may in the future redo this in more general terms with networkx, but at present it looks like
networkx does not really provide anything and requires a new dependency.
"""


class SignalGraph(set):
    """
    A SignalGraph is a bipartite directed graph, where the two types of nodes are ("signal" and "function").
    Signals are given explicit names, and functions are referenced by the signals they consume/produce.  Each
    signal may be produced by only one function.
    """

    def __init__(self, graph_specifier):

        assert all(len(el)==2 for el in graph_specifier), \
            'Your graph specifier must be a set of 2-tuples'

        if isinstance(graph_specifier, dict):
            graph_dict = {_tuplefy_node(n): f for n, f in graph_specifier.iteritems()}
            graph_specifier = graph_dict.keys()
        else:
            graph_specifier = clean_graph(graph_specifier)
            graph_dict = None
        self._graph_dict = graph_dict

        for se, de in graph_specifier:
            self.add((se, de))

    def __getitem__(self, index):
        assert self._graph_dict is not None, 'You can only dereference a SignalGraph that was instantiated with a dict.'
        sanitized_index = _tuplefy_node(index)
        return self._graph_dict[sanitized_index]

    def reverse(self):
        return self.__class__({(d, s) for s, d in self})

    def get_predecessor_signals(self, signal, given_signals = None, _memo = None):
        """
        Get predecessor signals, including the provided signal.
        :param signal:
        :param given_signals:
        :param _memo:
        :return:
        """
        if given_signals is None:
            given_signals = set()

        if _memo is None:
            _memo = {sig: {sig} for sig in given_signals}
        if signal in _memo:
            return _memo[signal]

        producing_function = self.get_function_producing_signal(signal)
        immediate_predecessors = producing_function[0] if producing_function is not None else ()
        all_predecessors = {signal, }.union(given_signals)
        for pred_signal in immediate_predecessors:
            sub_predecessors = self.get_predecessor_signals(pred_signal, _memo=_memo)
            all_predecessors = all_predecessors.union(sub_predecessors)
        _memo[signal] = all_predecessors

        return all_predecessors

    def get_successor_signals(self, signal):
        return self.reverse().get_predecessor_signals(signal)

    def get_predecessor_graph(self, signal, given_signals = None):
        """ Return the graph of all functions leading to the signal """
        predecessor_signals = self.get_predecessor_signals(signal, given_signals = given_signals)
        # Return nodes if all their inputs and any of their outputs are part of the predecessor graph
        return self.filter_graph(lambda (src_signals, dest_signals):
            all(s in predecessor_signals for s in src_signals) and any(d in predecessor_signals for d in dest_signals)
            )

    def get_successor_graph(self, signal):
        """ Return the graph of all functions downstream from the signal """
        return self.filter_graph(lambda n: n in self.reverse().get_predecessor_graph(signal).reverse())

    def filter_graph(self, rule):
        """
        Return a new graph obtained by filtering the current graph by the given rule.  This preserves the dict entries
        if the SignalGraph was instantiated as a dictionary.
        :param rule: A function taking (src_signals, dest_signals) as an argument and returning a boolean indicating if
            it passes the filter test.
        :return: A SignalGraph which is a subset of the current graph.
        """
        if self._graph_dict is None:
            return SignalGraph({n for n in self if rule(n)})
        else:
            return SignalGraph({n: self[n] for n in self if rule(n)})

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
        return set().union(*(set(s+d) for s, d in self))

    def get_produced_signals(self):
        return set().union(*(set(d) for _, d in self))

    def get_input_signals(self):
        return self.get_all_signals().difference(self.get_produced_signals())

    def get_consumed_signals(self):
        return set.union(*(set(s) for s, _ in self))

    def get_output_signals(self):
        return self.get_all_signals().difference(self.get_consumed_signals())

    def get_subgraph(self, input_signals = None, output_signals = None):

        input_signals = self.get_input_signals() if input_signals is None else set(input_signals)
        output_signals = self.get_output_signals() if output_signals is None else set(output_signals)

        # Find all nodes required to compute outputs
        subgraph_nodes = set().union(*tuple(self.get_predecessor_graph(sig, given_signals = input_signals) for sig in output_signals))
        subgraph = self.filter_graph(lambda n: n in subgraph_nodes)

        assert set.isdisjoint(input_signals, subgraph.get_produced_signals()), \
            'Input signals %s were both provided and computed in the subgraph %s, so they are doubly-defined.' \
            % (list(subgraph.get_produced_signals().intersection(input_signals)), subgraph)

        # Note that the following assert can also fail because input_signals is a superset of the subgraph's input signals,
        # but this should be caught in the previous assert.
        assert subgraph.get_input_signals() == input_signals, \
            'The provided input signals: %s are not enough to compute the output signals %s in graph %s' \
            % (list(input_signals), list(output_signals), self)

        return subgraph

    def get_parallel_order(self):
        """
        Get the a list of sets of function indices, where the index of the first list indicates the order in which that function
        can be executed (all functions in the i'th list must wait until all functions in 0th to i-1th lists are complete).
        :return: a list<set<tuple<tuple<*str>, tuple<*str>>>>
        """
        known_signals = self.get_input_signals().copy()
        remaining_function_nodes = self.copy()
        parallel_levels = []
        while len(remaining_function_nodes)>0:
            this_level = set()
            this_layers_outputs = set()
            for node in remaining_function_nodes.copy():
                input_signals, output_signals = node
                if known_signals.issuperset(input_signals):
                    remaining_function_nodes.remove(node)
                    this_level.add(node)
                    this_layers_outputs.update(output_signals)
            assert len(this_layers_outputs)> 0, 'Your graph seems to have loops.'
            known_signals.update(this_layers_outputs)
            parallel_levels.append(this_level)
        return parallel_levels


def clean_graph(graph_specifier):
    return {_tuplefy_node(n) for n in graph_specifier}


def _tuplefy_node((src_signals, dest_signals)):
    return _tuplefy_singles(src_signals), _tuplefy_singles(dest_signals)


def _tuplefy_singles(edge_specifier):
    return edge_specifier if isinstance(edge_specifier, tuple) else (edge_specifier, )
