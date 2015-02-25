import pytest
from utils.graph_utils import SignalGraph, clean_graph

__author__ = 'peter'


def get_test_graph(graph = 'basic'):

    if graph == 'basic':
        # 'a' --p1-> 'b' --p2--> 'd' --p3--> 'e'
        # 'c' -------------^
        return SignalGraph({
            ('a', 'b'),
            (('b', 'c'), 'd'),
            ('d', 'e'),
            })
    elif graph == 'dict':
        return SignalGraph({
            ('a', 'b'): 'thing1',
            (('b', 'c'), 'd'): 'thing2',
            ('d', 'e'): 'thing3',
            })


def test_graph_cutting():

    # Unsure of appropriate constructor
    # Could use shorthand notation to remove 1-tuples
    g = get_test_graph()

    # Looking for some function get_subgraph such that:
    assert g.get_subgraph(output_signals = ['d']) == SignalGraph({('a', 'b'), (('b', 'c'), 'd')})
    assert g.get_subgraph(input_signals = ['d']) == SignalGraph({('d', 'e')})
    assert g.get_subgraph(input_signals = ['b', 'c']) == SignalGraph({(('b', 'c'), 'd'), ('d', 'e')})
    assert g.get_subgraph(input_signals = ['b', 'c'], output_signals =['d']) == SignalGraph({(('b', 'c'), ('d', ))})

    with pytest.raises(AssertionError):
        # The following should raise an error because src was specified but did not completely cut off the graph.
        g.get_subgraph(input_signals = ['a'], output_signals =['d'])

    with pytest.raises(AssertionError):
        # The following should probably raise an error because src was specified but did not completely cut off the graph.
        g.get_subgraph(input_signals = ['a', 'b', 'c'])


def test_parallel_ordering():

    g = get_test_graph()
    assert g.get_parallel_order() == [clean_graph({('a', 'b')}), clean_graph({(('b', 'c'), 'd')}), clean_graph({('d', 'e')})]


def test_dict_behaviour():

    g = get_test_graph('dict')

    assert g['a', 'b'] == 'thing1'
    assert g[('a', ), ('b', )] == 'thing1'
    assert g[('b', 'c'), 'd'] == 'thing2'
    assert g[('b', 'c'), ('d', )] == 'thing2'
    assert g.get_subgraph(input_signals = ['b', 'c'])['d', 'e'] == 'thing3'


if __name__ == '__main__':

    test_dict_behaviour()
    test_graph_cutting()
    test_parallel_ordering()
