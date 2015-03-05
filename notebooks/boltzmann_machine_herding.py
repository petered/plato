
# coding: utf-8

# In[1]:

# Boltzmann Machine Demo
#
# Here we compare the time it takes Gibbs sampling and Herded-Gibbs to converge to the 
# stationary distribution in a Boltzmann Machine.  Results show that
# - Gibbs converges at a rate of 1/sqrt(t), while Herded-Gibbs initially converges at 
#   a rate of 1/t, but then stops due to its biased sampling.
# - Running Herding updates as a single block rather than per-unit causes the Herding 
#   algorithm to stop converging further from the stationary distribution than it 
#   otherwise would.
# 
# This demo takes about 10 seconds to run for 1 million iterations.  To run the whole 
# thing, click Cell>Run All.  The [*] on the left indicates that a cell is still executing.


# In[2]:

# Settings
mag = 0.4          # Standard Deviation of weights
w_power = 1.       # Weights are raised to this power (sign is preserved)
n_steps = 1000000  # Number of iterations.  Note: Sequential (non-block) Gibbs does a complete round robin in EACH iteration.
n_dims = 18        # Number of nodes in the Boltzmann Machine
random_seed = None       


# ##### Source Code 
# - [experimental.boltzmann_sampling](http://localhost:8888/edit/experimental/boltzmann_sampling.py) 
# - [general.math](http://localhost:8888/edit/general/math.py) 

# In[3]:

import numpy as np
from matplotlib import pyplot as plt
from experimental.boltzmann_sampling import gibbs_sample_boltzmann, herded_sample_boltzmann,     compute_exact_boltzmann_marginals, generate_boltzmann_parameters
from general.math import cummean
get_ipython().magic(u'matplotlib inline')


# In[4]:

# Initialize Weights
rng = np.random.RandomState(random_seed)
weights, biases = generate_boltzmann_parameters(mag = mag, power = w_power, n_dims = n_dims, rng=rng)


# In[5]:

# Compute exact marginal probabilities
exact_marginals = compute_exact_boltzmann_marginals(weights, biases)


# In[6]:

# Get the Gibbs Stats
single_gibbs_records = gibbs_sample_boltzmann(weights, biases, n_steps, rng=rng, block = False)
block_gibbs_records = gibbs_sample_boltzmann(weights, biases, n_steps, rng=rng, block = True)


# In[7]:

# Get the Herding Stats
single_herded_records = herded_sample_boltzmann(weights, biases, n_steps, block = False)
block_herded_records = herded_sample_boltzmann(weights, biases, n_steps, block = True)


# In[8]:

# Compute Error curves
L1_error = lambda x, tar: np.mean(np.abs(x-tar), axis = 1)
single_gibbs_error = L1_error(cummean(single_gibbs_records, axis = 0), exact_marginals)
block_gibbs_error = L1_error(cummean(block_gibbs_records, axis = 0), exact_marginals)
single_herded_error = L1_error(cummean(single_herded_records, axis = 0), exact_marginals)
block_herded_error = L1_error(cummean(block_herded_records, axis = 0), exact_marginals)


# In[9]:

# Plot (Warning - can be slow due to huge number of points)
from plotting.fast import fastloglog
plt.figure()
fastloglog(single_gibbs_error)
fastloglog(block_gibbs_error)
fastloglog(single_herded_error)
fastloglog(block_herded_error)
plt.loglog([1, n_steps], [1, n_steps**-1])
plt.loglog([1, n_steps], [1, n_steps**-.5])
plt.legend(['Gibbs', 'Block-Gibbs', 'Herding', 'Block-Herding', '1/x', '1/sqrt(x)'], loc='best')
plt.show()

