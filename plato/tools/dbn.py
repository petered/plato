from utils.graph_utils import SignalGraph

__author__ = 'peter'


def test_graph_utils():

    # 'a' --p1-> 'b' --p2--> 'd' --p3--> 'e'
    # 'c' -------------^
    g = SignalGraph({
        ('a', ): ('b', ),
        ('b', 'c'): ('d', )
        ('d' ): ('e', )
        })
    # Unsure of appropriate constructor
    # Could use shorthand notation to remove 1-tuples

    # Looking for some function get_subgraph such that:

    assert set(get_subgraph(g, dest = ['d']).nodes()) == {('a', ): ('b', ), ('b', 'c'): ('d', )}
    assert set(get_subgraph(g, src = ['d']).edges()) == {('d', ): ('e', )}
    assert set(get_subgraph(g, src = ['b', 'c']).edges()) == {('b', 'c'): ('d', ), ('d', ): ('e', )}
    assert set(get_subgraph(g, src = ['b', 'c'], dest =['d']).edges()) == {('b', 'c'): ('d', )}


    with pytest.raises(AssertionError):
        # The following should probably raise an error because src was specified but did not completely cut off the graph.
        get_subgraeph(g, src = ['a'], dest =['d']).edges())
