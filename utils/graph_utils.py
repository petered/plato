from collections import OrderedDict
from general.should_be_builtins import bad_value
from collections import Counter

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

        all_dest_notes = sum([d for _, d in graph_specifier], ())
        multiply_written_dest_nodes = [k for k, v in Counter(all_dest_notes).iteritems() if v > 1]
        assert len(multiply_written_dest_nodes) == 0, 'Notes %s are written to more than once.\nGraph: %s' \
            % (multiply_written_dest_nodes, graph_specifier)

        self._graph_dict = graph_dict

        for se, de in graph_specifier:
            self.add((se, de))

    def __getitem__(self, index):
        assert self._graph_dict is not None, 'You can only dereference a SignalGraph that was instantiated with a dict.'
        sanitized_index = _tuplefy_node(index)
        return self._graph_dict[sanitized_index]

    def items(self):
        assert self._graph_dict is not None, 'You can only call iteritems on a SignalGraph that was instantiated with a dict.'
        return self._graph_dict.items()

    def is_graph_dict(self):
        return self._graph_dict is not None

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
            assert len(functions_producing_signal) == 1, 'Invalid Graph: You cannot have 2 functions producing the same signal. %s' \
                % (self, )
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

        input_signals = self.get_input_signals() if input_signals is None else set(_tuplefy_singles(input_signals))
        output_signals = self.get_output_signals() if output_signals is None else set(_tuplefy_singles(output_signals))

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
        Get the a list of sets of function nodes, where the index of the first list indicates the order in which that function
        can be executed (all functions in the i'th list must wait until all functions in 0th to i-1th lists are complete).
        :return: a list<SignalGraph>
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
            parallel_levels.append(self.filter_graph(lambda n: n in this_level))
        return parallel_levels

    def get_serial_order(self):
        """
        Get a list/OrderedDict of function nodes.  Executing the graph in this order guarantees that all signals are
        computed before they become inputs to another node.  This order can be partly arbitrary because within a parallel
        level it doesn't matter which nodes are executed, so don't expect this to always return the same order for a
        given graph.
        :return: A list<*tuple<tuple<*str>, tuple<*str>>> if this is a graph_dict, else a
            OrderedDict<*tuple<tuple<*str>, tuple<*str>>:object>
        """
        parallel_order = self.get_parallel_order()

        if self.is_graph_dict():
            return OrderedDict(sum([level.items() for level in parallel_order], []))
        else:
            return sum([list(level) for level in parallel_order], [])


class FactorGraph(object):

    def __init__(self, variables, factors):
        """
        :param variables: A dict<int: function>, where function takes N arguments (one for each
            factor feeding into it, and procuces one output.
        :param factors: A dict<(int, int): reversable_function), Where function take one argument,
            produces one output, and has a "reverse" method, which implements the function in
            the other direction.

        * Note - we don't currently deal with the fact that we can't specify the order in which factors
        feed variables - we will have to do this at some point.  Maybe.
        :return:
        """
        assert all(src in variables and dest in variables for src, dest in factors.viewkeys()), \
            'All factors must link variables'
        self._variables = variables
        self._factors = factors

    def get_inference_path(self, variable_path):
        """
        Given a path defined in terms of variables int the factor graph, return a full inference that defines the order
        in which to compute factors and sample variables.

        :param varible_path: A list of 2-tuples, identifying the source, destination layers for the update.
        :return: path: An OrderedDict<(tuple<*int>, int): function> indicating the path to take.
        """

        variable_path = [(_tuplefy_singles(src), _tuplefy_singles(dest)) for src, dest in variable_path]

        path = OrderedDict()
        for src_vars, dest_vars in variable_path:
            for src_var in src_vars:
                for dest_var in dest_vars:
                    path[(src_var, ), factor_name(src_var, dest_var)] = \
                        self._factors[src_var, dest_var] if (src_var, dest_var) in self._factors else \
                        self._factors[dest_var, src_var].reverse if (dest_var, src_var) in self._factors else \
                        bad_value((src_var, dest_var), 'Factor %s does not exist in the graph: %s' % ((src_var, dest_var), self._factors.keys()))
            for dest_var in dest_vars:
                path[tuple(factor_name(src_var, dest_var) for src_var in src_vars), dest_var] = self._variables[dest_var]
        return InferencePath(path)

    def get_variable_path_from_io(self, input_signals, output_signals):

        input_signals = _tuplefy_singles(input_signals)
        output_signals = _tuplefy_singles(output_signals)
        outputs_needing_calculation = tuple(os for os in output_signals if os not in input_signals)
        direct_linking_factors = [(src, dest) for src, dest in self._factors if src in input_signals and dest in outputs_needing_calculation]
        outputs_of_direct_links = [dest for _, dest in direct_linking_factors]

        if set(outputs_needing_calculation).issubset(set(outputs_of_direct_links)):
            variable_path = [(tuple(src for src, dest in direct_linking_factors if dest==out), out) for out in outputs_needing_calculation]
            return variable_path
        else:
            # Ugh, so we have to deal with loops and all this stuff - lets pospone.
            raise NotImplementedError('Have not yet implemented automatic path finding between src, dest nodes.  Do it yourself!')

    def get_input_variables(self):
        written_varibles = set(dest for _, dest in self._factors)
        return {v_name: v for v_name, v in self._variables.iteritems() if not v_name in written_varibles}


def _clean_path(specified_path):
    """
    :param specified_path: Defines the flow of information along a graph.  Generally, it should be of the form:
        [..., ((in_sig_0, in_sig_1, ...): out_sig), ...] where in_sig, out_sig are int/strings identifying signals.
    :return:
    """

    assert isinstance(specified_path, list) and all(len(p)==2 for p in specified_path), 'Path must be a list of 2-tuples.  We got "%s".' % (specified_path, )
    assert all(isinstance(src, (tuple, list, int, str)) and isinstance(dest, (int, str)) for src, dest in specified_path)
    path = []
    for srcs, dests in specified_path:
        srcs = (tuple(srcs) if isinstance(srcs, (tuple, list)) else (srcs, ))
        dests = (tuple(dests) if isinstance(dests, (tuple, list)) else (dests, ))
        for dest in dests:
            path.append((srcs, dest))
    return path


def factor_name(src_node, dest_node):
    return 'f[%s,%s]' % (src_node, dest_node)


def clean_graph(graph_specifier):
    return {_tuplefy_node(n) for n in graph_specifier}


def _tuplefy_node((src_signals, dest_signals)):
    return _tuplefy_singles(src_signals), _tuplefy_singles(dest_signals)


def _tuplefy_singles(edge_specifier):
    return edge_specifier if isinstance(edge_specifier, tuple) else (edge_specifier, )


def _singlefy_tuples(edge_specifier):
    if isinstance(edge_specifier, (list, tuple)):
        assert len(edge_specifier)==1, 'You tried to singlefy an edge specifier, did not have 1 signal: %s' % (edge_specifier, )
        singlified, = edge_specifier
    else:
        singlified = edge_specifier
    return singlified


class InferencePath(object):

    def __init__(self, specified_path):
        """
        :param specified_path: An OrderedDict<(*int/str,int/str): func>
        """

        assert isinstance(specified_path, OrderedDict) and all(len(p)==2 for p in specified_path), \
            'Path must be an OrderedDict mapping of 2-tuples to functions.  We got "%s".' % (specified_path, )
        assert all(isinstance(src, (tuple, list, int, str)) and isinstance(dest, (int, str, list, tuple)) for src, dest in specified_path)
        self._path = OrderedDict(((_tuplefy_singles(src), _singlefy_tuples(dest)), f) for (src, dest), f in specified_path.iteritems())
        self._required_inputs = SignalGraph(self._path).get_input_signals()

    def execute(self, input_signal_dict):

        assert set(input_signal_dict.keys()).issuperset(self._required_inputs), 'The inputs you provided: %s, did not match the set of required inputs: %s' \
            % (input_signal_dict.keys(), set(self._required_inputs))

        signals = input_signal_dict.copy()
        for (srcs, dest), func in self._path.iteritems():
            out = func(*[signals[src] for src in srcs])
            signals[dest] = out
        return signals
