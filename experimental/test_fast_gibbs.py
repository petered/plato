__author__ = 'peter'


"""
Gibbs sampling can be really slow in python because it can't just be turned in a bit numpy vector operation,
since each update of each unit depends on the last.  So here we experiment with different thing to make it
faster.
1) Plain old python
2) Scipy weave
3) Theano scan op
"""


def test_fast_gibbs():

