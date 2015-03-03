# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
from matplotlib import pyplot as plt

# <codecell>

# Setup functions
binary_count_matrix = lambda n_bits: np.right_shift(np.arange(2**n_bits)[:, None], np.arange(n_bits-1, -1, -1)[None, :]) & 1

sigm = lambda x: 1/(1+np.exp(-x))

L1_error = lambda x, tar: np.mean(np.abs(x-tar), axis = 1)

def cummean(x, axis):
    x=np.array(x)
    normalized = np.arange(1, x.shape[axis]+1).astype(float)[(slice(None), )+(None, )*(x.ndim-axis-1)]
    return np.cumsum(x, axis)/normalized

def random_symmetric_mat(mag, power, n_dims, rng):
    w = mag*np.random.randn(n_dims, n_dims)
    w = 0.5*(w+w.T)*(1-np.diag(np.ones(n_dims)))
    w = np.sign(w)*np.abs(w)**power
    return w

# <codecell>

# Settings
mag = 0.4        # a.k.a. la
w_power = 1.     # a.k.a. a
n_steps = 500000 # a.k.a. T
n_dims = 12      # a.k.a. N
seed = None

# <codecell>

# Initialize Weights
rng = np.random.RandomState(seed)
biases = mag*np.random.randn(n_dims)
weights = random_symmetric_mat(mag = mag, power = w_power, n_dims = n_dims, rng = rng)
assert np.all(weights[np.arange(n_dims), np.arange(n_dims)]==0)
assert np.array_equal(weights, weights.T)

# <codecell>

# Compute exact marginal probabilities
def exact_marginals_func(weights, biases):
    """
    weights is a (n_nodes, n_nodes) symmetric matrix
    biases is a (n_nodes, ) vector
    """
    # Lets do this memory-intensive but fast
    n_nodes = weights.shape[0]
    assert n_nodes == len(biases) == weights.shape[1]
    assert np.allclose(weights, weights.T)
    bmat = binary_count_matrix(n_nodes).astype(float)  # (2**n_dims, n_dims)
    energy = 0.5*np.einsum('ij,jk,ik->i', bmat, weights, bmat) + (bmat*biases).sum(axis=1)  # (2**n_dims, )
    
    exp_log_prob = np.exp(energy)  # (2**n_dims, )  # Negative sign?
    q = np.sum(exp_log_prob[:, None]*bmat, axis = 0)  # (n_dims, )
    normalizer = np.sum(exp_log_prob)  # Scalar
    marginals = q/normalizer
    return marginals
exact_marginals = exact_marginals_func(weights, biases)

# <codecell>

# Get the Gibbs Stats
def gibbs_sample(weights, biases, n_steps, rng, block = False):
    """
    weights is a (n_nodes, n_nodes) symmetric matrix
    biases is a (n_nodes, ) vector
    n_steps is the number of steps to run Gibbs sampling for 
    rng is a random number generator
    
    returns records: A (n_steps, n_nodes) array indicating the state of the MCMC at each step. 
    """
    n_dims = weights.shape[0]
    n_samples = len(biases)
    assert n_dims == len(biases) == weights.shape[1]
    records = np.empty((n_steps, n_dims))
    x = 0.5 > rng.rand(n_dims)
    for t in xrange(n_steps):
        if block:
            x = sigm(x.dot(weights)+biases) > rng.rand(n_dims)
        else:
            for i in xrange(n_dims):
                x[i] = sigm(x.dot(weights[i, :])+biases[i]) > rng.rand()
        records[t, :] = x
    return records  
single_gibbs_records = gibbs_sample(weights, biases, n_steps, rng=rng, block = False)
block_gibbs_records = gibbs_sample(weights, biases, n_steps, rng=rng, block = True)

# <codecell>

# Get the Herding Stats
def herded_sample(weights, biases, n_steps, rng, block = False):
    """
    weights is a (n_nodes, n_nodes) symmetric matrix
    biases is a (n_nodes, ) vector
    n_steps is the number of steps to run Gibbs sampling for 
    rng is a random number generator
    
    returns records: A (n_steps, n_nodes) array indicating the state of the MCMC at each step. 
    """
    n_dims = weights.shape[0]
    n_samples = len(biases)
    assert n_dims == len(biases) == weights.shape[1]
    records = np.empty((n_steps, n_dims))
    x = 0.5 > rng.rand(n_dims)
    h = np.zeros(x.shape)
    for t in xrange(n_steps):
        if block:
            p = sigm(x.dot(weights)+biases)
            h += p
            x = h > 0.5
            h -= x
        else:
            for i in xrange(n_dims):
                p = sigm(x.dot(weights[i, :])+biases[i])
                h[i] += p
                x[i] = h[i] > 0.5
                h[i] -= x[i]
        records[t, :] = x
    return records  
single_herded_records = herded_sample(weights, biases, n_steps, rng=rng, block = False)
block_herded_records = herded_sample(weights, biases, n_steps, rng=rng, block = True)

# <codecell>

# Compute Error curves
single_gibbs_error = L1_error(cummean(single_gibbs_records, axis = 0), exact_marginals)
block_gibbs_error = L1_error(cummean(block_gibbs_records, axis = 0), exact_marginals)
single_herded_error = L1_error(cummean(single_herded_records, axis = 0), exact_marginals)
block_herded_error = L1_error(cummean(block_herded_records, axis = 0), exact_marginals)

# <codecell>

# Plot
plt.figure()
plt.loglog(single_gibbs_error)
plt.loglog(block_gibbs_error)
plt.loglog(single_herded_error)
plt.loglog(block_herded_error)
plt.loglog([1, n_steps], [.5, n_steps**-1])
plt.loglog([1, n_steps], [.5, n_steps**-.5])
plt.legend(['Gibbs', 'Block-Gibbs', 'Herding', 'Block-Herding', '1/x', '1/sqrt(x)'], loc='best')
plt.show()

# <codecell>


# <codecell>


