# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
from matplotlib import pyplot as plt

# <codecell>

# Setup functions
binary_count_matrix = lambda n: ((np.arange(2**n)[:, None] >> np.arange(n-1, -1, -1)[None, :]).astype(bool) & True)

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

mag = 0.4      # la
w_power = 1.   # a
n_steps = 50000  # T
n_dims = 18   # N
seed = None

# <codecell>

rng = np.random.RandomState(seed)
biases = mag*np.random.randn(n_dims)
weights = random_symmetric_mat(mag = mag, power = w_power, n_dims = n_dims, rng = rng)

# <codecell>

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
    inner_sum = 0.5*np.einsum('ij,jk,ik->i', bmat, weights, bmat) + (bmat*biases).sum(axis=1)  # (2**n_dims, )
    
    exp_log_prob = np.exp(inner_sum)  # (2**n_dims, )
    q = np.sum(exp_log_prob[:, None]*bmat, axis = 0)  # (n_dims, )
    normalizer = np.sum(exp_log_prob)  # Scalar
    marginals = q/normalizer
    return marginals
exact_marginals = exact_marginals_func(weights, biases)

# <codecell>

def gibbs_fc(weights, biases, n_steps, rng, block = False):
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
                x[i] = sigm(x.dot(weights[i])+biases[i]) > rng.rand() 
        records[t, :] = x
    return records  

single_gibbs_records = gibbs_fc(weights, biases, n_steps, rng=rng, block = False)
block_gibbs_records = gibbs_fc(weights, biases, n_steps, rng=rng, block = True)

# <codecell>

# Compute Error curves
TARGET = 'gibbs-final'
target_distribution = {
    'gibbs-final': np.mean(single_gibbs_records, axis = 0),
    'exact': exact_marginals
    }[TARGET]

single_gibbs_error = L1_error(cummean(single_gibbs_records, axis = 0), target_distribution)
block_gibbs_error = L1_error(cummean(block_gibbs_records, axis = 0), target_distribution)

# <codecell>

# Plot
plt.figure()
plt.loglog(single_gibbs_error)
plt.loglog(block_gibbs_error)
plt.show()

# <codecell>


