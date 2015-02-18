# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
import matplotlib.pyplot as plt
from numpy import convolve
from IPython import display
INLINE = True
plt.ion()
if INLINE:
    %matplotlib inline
    plt.ion()
    def redraw(figure):
        display.clear_output(wait=True)
        display.display(figure)
else:
    def redraw(figure):
        plt.draw()    

# <codecell>

# Set Parameters
n_time_steps = 1000
kernel_len = 40
n_filters = 10
waveform_data = 'sinusoid'
w_init_mag = 0.01
w_L2_norm = 0.0003
sigma = 1.
sigma_lower_bound = 0.05
n_gibbs = 1
eta = 0.0005
n_training_steps = 5000
plot_every = 40
print_every = 1000
persistent = False
seed = None

# <codecell>

# Create data
rng = np.random.RandomState(seed)
t = np.arange(n_time_steps)
data = {
    'sinusoid': lambda: np.sin(t/10.)+np.sin(t/(2*np.pi)),
    'sawtooth': lambda: (t/10.)%10 + (t/7.)%7-1
}[waveform_data]()

# <codecell>

# Define Functions
convup = lambda x, w: np.array([convolve(x, w_row, 'valid') for w_row in w])
convdown = lambda x, w: np.sum([convolve(x_row, w_row[::-1], 'full') for x_row, w_row in zip(x, w)], axis = 0)
sigmoid = lambda x: 1./(1+np.exp(-x))
sample_gaussian = lambda mean, sigma: mean+np.random.randn(*mean.shape)*sigma
sample_bernoulli = lambda x: x > rng.rand(*x.shape)
propup = lambda x: sample_bernoulli(sigmoid(convup(x, w) + c[:, None]))
propdown = lambda h: sample_gaussian(convdown(h, w) + b, sigma)
compute_w_grad = lambda x, h: np.array([convolve(x, h_row[::-1], 'valid') for h_row in h])/n_time_steps
compute_b_grad = lambda v: np.mean(v)
compute_c_grad = lambda h: np.mean(h, axis = 1)
compute_sigma_grad = lambda v, h: np.mean((v-b)**2 - 2*v*convdown(h, w))/ (sigma**3)
get_state_string = lambda: 'step: %s: w-mean: %.2g, w-std: %.2g, sigma: %.2g, b: %.2g, c_mean: %.2g, c_std: %.2g' %(i, np.mean(w), np.std(w), sigma, b, np.mean(c), np.std(c))

# <codecell>

# Train
b = np.zeros(())
c = np.zeros(n_filters)
w = w_init_mag*np.random.randn(n_filters, kernel_len)

training_fig = plt.figure()
plt.subplot(3,1,1)
plt.plot(data)
recon_plot, = plt.plot(data)
plt.subplot(3,1,2)
hidden_plot = plt.imshow(propup(data), cmap = 'gray', interpolation = 'nearest', aspect = 10)
plt.subplot(3,1,3)
w_plot = plt.imshow(w, cmap = 'gray', interpolation = 'nearest')
plt.colorbar()
plt.show()

print 'Training...'
h_sleep = None
for i in xrange(n_training_steps):
    v_wake = data
    h_wake = propup(v_wake)
    h_sleep = h_wake if (not persistent or h_sleep is None) else h_sleep
    for _ in xrange(n_gibbs):
        v_sleep = propdown(h_sleep)
        h_sleep = propup(v_sleep)
    w += eta*(compute_w_grad(v_wake, h_wake) - compute_w_grad(v_sleep, h_sleep)) - w_L2_norm*w
    b += eta*(compute_b_grad(v_wake) - compute_b_grad(v_sleep))
    c += eta*(compute_c_grad(h_wake) - compute_c_grad(h_sleep))
    sigma += eta*(compute_sigma_grad(v_wake, h_wake) - compute_sigma_grad(v_sleep, h_sleep))
    sigma = np.maximum(sigma, sigma_lower_bound)
    if i%plot_every == 0:
        print get_state_string()
        recon_plot.set_ydata(convdown(h_sleep, w) + b)
        hidden_plot.set_array(h_sleep)
        w_plot.set_array(w)
        redraw(training_fig)
print '...Done.'

# <codecell>

# Free-sampling
free_sampling_fig = plt.figure()
n_steps = 1000
visible_noiseless_sample = visible_sample = np.random.randn(*data.shape)
free_sampling_plot, = plt.plot(visible_noiseless_sample)
free_sampling_fig.show()
for _ in xrange(n_steps):
    hidden_sample = propup(visible_sample)
    visible_noiseless_sample = convdown(hidden_sample, w) + b
    visible_sample = sample_gaussian(visible_noiseless_sample, sigma)
    free_sampling_plot.set_ydata(visible_noiseless_sample)
    redraw(free_sampling_fig)

