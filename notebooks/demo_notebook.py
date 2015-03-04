# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import matplotlib.pyplot as plt
import numpy as np

# <codecell>

# Make Data
x = np.linspace(0, 4, 200)
y = -(x-4)*(x+1)

# <codecell>

# Plot
plt.plot(x, y)
plt.xlabel('Years Spend Making Productivity Tools')
plt.ylabel('Productivity')
plt.show()

# <codecell>


