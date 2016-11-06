import numpy as np

__author__ = 'peter'

sigm = lambda x: 1./(1+np.exp(-x))


bernoulli = lambda k, p: (p**k)*((1-p)**(1-k))  # Maybe a not the fastest way to do it but whatevs

