"""
    This script plots the scaling of the cost with the number of channels
-------------------------------------------------------------------------------
created on:
    Tue 25 Feb 2025
-------------------------------------------------------------------------------
last change:
    Tue 25 Feb 2025
-------------------------------------------------------------------------------
notes:
-------------------------------------------------------------------------------
contributors:
    Jose:
        name:       Jose Betancourt
        email:      jose.betancourtvalencia@yale.edu
-------------------------------------------------------------------------------
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from numba import njit

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 15,
    "font.serif": ["Garamond"],
})

colormap = 'gist_rainbow'

#------------------------------------------------------------------------------
# SCALING WITH NUMBER OF CHANNELS
#------------------------------------------------------------------------------
@njit
def cost(x,w,a):
    return (1-x)*(1+a*x/(w**2+(1-x)**2))

# Define parameters
x_vals = np.linspace(0.,1.,100)
N_plots = 11
w_min, w_max = 0.2, 1.2
w_vals = np.linspace(w_min, w_max, N_plots)
a = 1.

# Create plot
fig = plt.figure(figsize=(6.4, 4.8*(1.6/2)))
csfont = {'fontsize':15}

# Colormap is created
cmap = plt.get_cmap(colormap, N_plots)
color = [cm.get_cmap(cmap)(x)[:3] for x in range(N_plots)]
norm = mpl.colors.Normalize(vmin=w_min-0.5*(w_max-w_min)/(N_plots-1), vmax=w_max+0.5*(w_max-w_min)/(N_plots-1))
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

ax1 = fig.add_subplot(111)
# Plot for each value of omega
for i, w in enumerate(w_vals):
    ax1.plot(x_vals, cost(x_vals,w,a), c=color[i], lw=2)
    
ax1.set_ylabel('Energetic cost ${\cal C}_N(\omega)/{\cal C}_0(\omega)$')
ax1.set_xlabel('Number of channels $N/N_s$')

cbar = fig.colorbar(sm, ticks=[w for j, w in enumerate(w_vals) if j%2==0])
cbar.set_label('$\omega \\tau_{RC}$', **csfont)
cbar.ax.tick_params(labelsize=15)

plt.savefig('../figures/cost_scaling.pdf', dpi=600, bbox_inches='tight')
plt.show()
