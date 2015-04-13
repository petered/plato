import numpy as np
from abc import abstractmethod
from plato.interfaces.decorators import symbolic_stateless, symbolic_standard
from plato.interfaces.helpers import get_theano_rng
from plato.tools.basic import softmax
from theano.sandbox.cuda.rng_curand import CURAND_RandomStreams
from theano.sandbox.rng_mrg import MRG_RandomStreams
import theano.tensor as tt
from theano.tensor.elemwise import TensorVariable

__author__ = 'peter'


def sample_categorical(rng, p, axis = -1, values = None):
    """
    p is a n-d array, where the final dimension is a discrete distibution (does not need to be normalized).
    Sample from that distribution.
    This will return an array of shape p.shape[:-1] with values in range [0, p.shape[-1])

    :param rng: A theano shared_randomstreams.RandomStream object
    :param p: An ndarray of arbitrary shape, where the values along (axis) are interpreted as an unnormalized
        discrete probability distribution (so if p.shape[2]==5, it means that the variable can take on 5 possible
        values).
    :param axis: The axis which we consider to be the distribution (only -1 (last axis)) supported now.
    :param values: The values of the variable.  len(values) must equal p.shape[axis].  If not included, the
        values will be considered to be integers in range(0, p.shape[axis])
    """
    # TODO: assert no negative values in p / assert p normalized along axis instead of dividing
    # TODO: assert len(values) == p.shape[axis]
    assert axis == -1, 'Currenly you can only sample along the last axis.'
    p = p/tt.sum(p, axis = axis, keepdims=True)
    # TODO: Check that differnt RNGs are doing the same thing!

    if isinstance(rng, TensorVariable):
        # Externally generated random numbers - we receive the maximum number of uniform random numbers
        # we could need, and then generate samplews from thos.
        old_p_shape = p.shape
        random_numbers = rng[:p.size].reshape((p.size, 1))
        cumulative_prob_mass = tt.cumsum(p.reshape((-1, p.shape[-1])), axis = 1)
        samples = random_numbers < cumulative_prob_mass
        samples.reshape(old_p_shape)
    elif isinstance(rng, MRG_RandomStreams):
        # MRG_RandomStreams is faster but only works for 2-d pvals, so we have to reshape and
        # then unreshape.
        old_p_shape = p.shape
        samples = rng.multinomial(n=1, pvals = p.reshape((-1, p.shape[-1])))
        samples = samples.reshape(old_p_shape)
    elif isinstance(rng, CURAND_RandomStreams):
        # TODO: Make this work if possible - problem now is it needs to know shape in advance
        raise NotImplementedError("Curand doesn't work yet.")
        cumulative_prob_mass = np.cumsum(p, axis = axis)
        samples = rng.uniform(size = tt.set_subtensor(p.shape[axis], 1)) > cumulative_prob_mass
    else:
        samples = tt.switch(tt.eq(p.size, 0), tt.zeros(p.shape), rng.multinomial(n=1, pvals = tt.switch(tt.eq(p.size, 0), 1, p)))

    indices = tt.argmax(samples, axis = -1)  # Argmax is just a way to find the location of the only element that is 1.
    if values is not None:
        return values[indices]
    return indices


bernoulli_likelihood = lambda k, p: tt.switch(k, p, 1-p)  # or (p**k)*((1-p)**(1-k))



@symbolic_stateless
def p_x_given(x, w, y, **p_w_given_kwargs):
    """
    Note that we can just switch around x and w if we want to sample x's.

    If we swap the shapes in p_w_given:
    x -> w.T (nY, nX)
    w -> x.T (nX, nS)
    y -> y.T (nY, nS)

    A point (i, j) in alpha now indicates input-dimension-i, sample-j

    We now interpret the result as an array of probabilities of x[alpha] taking on each value.
    """

    p_x = p_w_given(x=w.T, w=x.T, y=y.T, **p_w_given_kwargs)
    return p_x  # (n_alpha, n_possible_ws) or (n_alpha, ) for boolean_ws



@symbolic_stateless
def p_w_given(x, w, y, alpha=None, possible_vals = (0, 1), binary = True, prior = None, input2prob = tt.nnet.sigmoid):
    """
    We have the following situation:
    y ~ Bernoulli(sigm(x.dot(w)))
    Where:
        x is a matrix of size (nS, nX)
        w is a booloean matrix of size (nX, nY)
        Y is a matrix of size (nS, nY)

    We want to find the probabilites of elements w[alpha] given X, Y, where
    alpha is some set of indices in w.

    :param x: A shape (nS, nX) tensor
    :param y: A shape (nS, nY) tensor
    :param alpha: Some indices that we will use to reference the elements of w that we want to modify.
    :param possible_vals: Possible states of w.
    :param binary: True if you want to treat your weights as Bernoulli variables.  False if, in the more general
        case, you want to treat them as categorical variables.   This affects the shape of the output.
    :return: An array of probabilities of w[alpha] taking on each value.  Its shape depends on boolean_ws:
        If boolean_ws is True, the shape is (n_alpha, )
        If boolean_ws is False, the shape is (n_alpha, len(possible_ws))
    """
    if binary: assert len(possible_vals) == 2, 'Come on.'
    if prior is None:
        prior = np.ones(len(possible_vals))/len(possible_vals)
    else:
        assert len(prior) == len(possible_vals)

    v_alpha_wk = compute_hypothetical_vs(x, w, alpha, possible_vals=possible_vals)  # (nS, n_alpha, n_possible_ws)
    y_alpha = y[:, alpha[1]] if alpha is not None else y[:, tt.arange(w.size) % y.shape[1]]
    log_likelihood = tt.sum(tt.log(bernoulli_likelihood(y_alpha[:, :, None], input2prob(v_alpha_wk))), axis = 0) + tt.log(prior)  # (n_alpha, n_possible_ws)
    if binary:
        # Note: Possibly reformulating this would let theano do the log-sigmoid optimization.  Would be good to check.
        p_w_alpha_w1 = tt.nnet.sigmoid(log_likelihood[:, 1] - log_likelihood[:, 0])  # (n_alpha, )
        return p_w_alpha_w1
    else:
        p_w_alpha_wk = softmax(log_likelihood, axis = 1)
        return p_w_alpha_wk


@symbolic_stateless
def compute_hypothetical_vs(x, w, alpha=None, possible_vals = (0, 1)):
    """
    We have v = x.dot(w)
    Where w can take on a discrete set of states.  We want to ask:
    "Suppose we change w[i, j], where (i, j) are indices in some set in alpha.  What is the value of v[j] for each of the
    possible values of w that we could change it to?"

    :param x: A shape (nS, nX) tensor
    :param w: A shape (nX, nY) tensor
    :param alpha: A 2-tuple of indices of w to reference.
    :return: v_alpha_wk: An array of shape (nS, n_alpha, len(possible_ws)) of "would be" values of v[j] - that is, the
        value v_alpha_wk[s, j, k] is what v[s, alpha[j]] WOULD be if we changed w[alpha_rows[j],alpha_cols[j]] to possible_ws[k]
    """
    v_current = x.dot(w)  # (nS, nY)
    # Each potential change in w changes a column in v_current.  Lets get the set of vs that are affected by each
    # selected weight w[alpha]
    if alpha is not None:
        alpha_rows, alpha_cols = alpha
        v_current_alpha = v_current[:, alpha_cols]  # (nS, n_alpha)
        x_alpha = x[:, alpha_rows]  # (n_S, n_alpha)
        v_alpha_0 = v_current_alpha - x_alpha * w[alpha]  # (nS, n_alpha) - What v would be if the input x for each alpha were set to zero.
        v_alpha_wk = v_alpha_0[:, :, None] + x_alpha[:, :, None] * possible_vals  # (nS, n_alpha, n_possible_ws) - What v would be if the input x for each alpha were set to each possible value of w.
    else:
        v_alpha_0 = v_current[:, None, :] - x[:, :, None] * w[None, :, :]  # (nS, nX, nY): Current with each weight zeroed
        v_alpha_wk = (v_alpha_0[:, :, :, None] + x[:, :, None, None] * possible_vals).reshape((x.shape[0], -1, len(possible_vals)))  # (nS, w.size, n_possible_ws)
    return v_alpha_wk  # (nS, n_alpha, n_possible_ws)


def gibbs_sample(p_wa, rng):
    return rng.choice(p_wa)


class IIndexGenerator(object):

    @abstractmethod
    def __call__(self):
        """
        Returns the next set of indices, and whatever state updates may be required.

        return: indices, updates   where:
            indices is a tuple that will be used to dereference an array
            updates contains whatever state updates are required.
        """

@symbolic_standard
class BaseIndexGenerator(IIndexGenerator):

    def __init__(self, size):
        self._size = size
        self._n_total = size if isinstance(size, int) else np.prod(size)

    def __call__(self):
        (vector_ixs, ), updates = self._get_vector_indices_and_updates()
        full_indices = \
            (vector_ixs, ) if isinstance(self._size, int) else \
            ind2sub(vector_ixs, self._size)
        return full_indices, updates

    @abstractmethod
    def _get_vector_indices_and_updates(self):
        pass


class RandomIndexGenerator(BaseIndexGenerator):
    """
    Randomly select some subset of the indices out of the full set.
    """

    def __init__(self, size, n_indices, seed = None):
        BaseIndexGenerator.__init__(self, size)
        self._n_indices = n_indices
        self._rng = get_theano_rng(seed)

    def _get_vector_indices_and_updates(self):
        ixs = self._rng.choice(size = self._n_indices, a = self._n_total, replace = False)
        return (ixs, ), []


class SequentialIndexGenerator(BaseIndexGenerator):

    def __init__(self, size, n_indices):
        BaseIndexGenerator.__init__(self, size)
        self._n_indices = n_indices

    def _get_vector_indices_and_updates(self):
        ixs = tt.shared(np.arange(self._n_indices))
        next_ixs = (ixs+self._n_indices) % self._n_total
        return (ixs, ), [(ixs, next_ixs)]


class OrderedIndexGenerator(BaseIndexGenerator):

    def __init__(self, order, n_indices, size = None):
        if size is None:
            size = len(order)
        BaseIndexGenerator.__init__(self, size)
        assert n_indices <= len(order)
        assert np.array_equal(np.sort(np.unique(order)), np.arange(len(order))), 'The order must be some permutation of indeces from 0 to max(order)'
        self._order = tt.constant(order)
        self._n_indices = n_indices

    def _get_vector_indices_and_updates(self):
        base_ixs = tt.shared(np.arange(self._n_indices))
        next_base_ixs = (base_ixs+self._n_indices) % self._n_total
        return (self._order[base_ixs], ), [(base_ixs, next_base_ixs)]


class RowIndexGenerator(OrderedIndexGenerator):
    """
    Return indeces of all elements within a row in a matrix
    """

    def __init__(self, size, n_rows_per_iter=1):

        assert len(size) == 2, 'This only works for 2-d indices'
        n_rows, n_cols = size
        order = np.arange(np.prod(size))
        OrderedIndexGenerator.__init__(self, order=order, n_indices = n_rows_per_iter*n_cols, size = size)

    #
    # def __call__(self):
    #
    #     base_ixs = tt.shared(np.arange(self._n_indices))
    #
    #     row_ixs =
    #
    #


# @symbolic_single_out_single_update
# class IndexGenerator(object):
#     """
#     In Gibbs sampling, we rotate the indices that we update.  This is the guy who decides who gets updated when.
#     """
#
#     def __init__(self, n_total, n_indices, randomization = 'every', rng = None, np_rng = None):
#
#         assert randomization in ('once', 'every', None)
#         assert n_indices <= n_total, "Can't sample more indices than the total number available!"
#         self._randomization = randomization
#         self._n_indices = n_indices
#         self._n_total = n_total
#         np_rng = np.random.RandomState(None) if np_rng is None else np_rng
#
#         self._rng = rng if rng is not None else RandomStreams(np_rng.randint(1e9))
#         if randomization == 'once':
#             # Need to use the np_rng because theano rng automatically updates whether we want it or not.
#
#             self._update_order = np_rng.choice(size = self._n_total, a = self._n_total, replace = False)
#
#     def __call__(self):
#
#         if self._randomization == 'every':
#             ixs = self._rng.choice(size = self._n_indices, a = self._n_total, replace = False)
#             ix_updates = []
#         else:
#             base_ixs = tt.shared(np.arange(self._n_indices))
#             next_base_indices = (base_ixs+self._n_indices) % self._n_total
#             ixs = base_ixs if self._randomization is None else self._update_order[base_ixs]
#             ix_updates = [(base_ixs, next_base_indices)]
#         return ixs, ix_updates
#
#
# @symbolic_standard
# class MatrixIndexGenerator(IIndexGenerator):
#
#     def __init__(self, ):
#         n_total = np.prod(shape)
#         self._shape = shape
#         self._base_index_generator = IndexGenerator(n_total = n_total, n_indices=nix(n_total, frac), randomization=randomization, rng = rng, np_rng=np_rng)
#
#     def __call__(self):
#         base_indices, updates = self._base_index_generator()
#         return ind2sub(base_indices, self._shape), updates


def nix(total, frac):
    """
    Tells you the number of indeces to update.  The formula is just
    """
    assert 0 < frac <= 1
    return int(np.ceil(float(frac) * total))


def ind2sub(ind, shape):
    assert len(shape) == 2, 'Only implemented for 2-d case, and only with pre-known shape'
    n_rows, n_cols = shape
    return ind / n_cols, ind % n_cols

