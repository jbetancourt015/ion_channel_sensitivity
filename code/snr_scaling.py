"""
    This script plots the scaling of the signal-to-noise ratio with the number
    of ion channels.
-------------------------------------------------------------------------------
created on:
    Wed 3 Dec 2025
-------------------------------------------------------------------------------
last change:
    Wed 3 Dec 2025
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
import logging

plt.rcParams.update({
    'text.usetex': False,  # keep LaTeX off globally
    'mathtext.fontset': 'cm',   # or 'stixsans' for sans-serif
    'mathtext.rm': 'Helvetica',
    'mathtext.it': 'Helvetica:italic',
    'mathtext.bf': 'Helvetica:bold',
    'font.family': 'Helvetica',
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'axes.linewidth': 0.5,   # thin axis lines for publication
    'pdf.fonttype': 42,      # ensures text stays as text in Illustrator
    'ps.fonttype': 42,       # same for PostScript
})

mpl.rcParams['figure.dpi'] = 300

logging.getLogger("fontTools").setLevel(logging.ERROR)
logging.getLogger("matplotlib.backends.backend_pdf").setLevel(logging.ERROR)

# Nature figure size
width = 3.5
height = 3.2

con_colors = np.array([[0, 77, 128], [181, 23, 0], [1, 113, 0], [242, 112, 0], 
                   [120, 0, 150], [0, 168, 157], [203, 41, 123], [0, 0, 0]])/255;
latex_gray = np.array([240, 240, 240])/255

colormap = 'cool_r'

import os
from params import kT, eps, sig, a, c, l_D, tau, R, n_points, N_log_min, N_log_max, FIGURES_DIR

# Derived quantities
tau_D = eps / a
om    = 1 / tau

# Crossover point
N_s = (1/8)*(a*(tau_D**2)/(tau*c*l_D))*((R/sig)**2)*(1+(om*tau)**2)

# N values
N_vals = np.logspace(N_log_min, N_log_max, n_points)

# SNR functions
def SNR_ratio(N):
    return N/(N_s+N)

# FIGURE: Signal-to-noise as a function of number of channels------------------
fig, ax = plt.subplots(figsize=(.9*width, .7*height))

ax.plot(N_vals, SNR_ratio(N_vals), c=con_colors[0], lw=2)
ax.plot([N_s,N_s], [SNR_ratio(1.),1.], c=con_colors[1], ls='--', lw=1)

# ax.tick_params(axis='y')
ax.text(2*N_s, 0.3, '$N^*$', c=con_colors[1])

ax.set_yscale('log')
ax.set_xscale('log')
ax.set_ylabel('$\\text{SNR}_N(\omega)/\\text{SNR}_{\\text{cell}}(\omega)$')
ax.set_xlabel('Number of channels ($N$)')

plt.savefig(os.path.join(FIGURES_DIR, 'snr_saturation.pdf'), dpi=600, bbox_inches='tight')
plt.show()

# Numerical results used in the text
S_shot = (np.pi/4)*((sig**2)*(tau_D**2)*a*kT)/l_D
S_JN = 4*(np.pi**1.5)*(sig**3)*(c**2)*kT/a
V0 = np.sqrt((1/(16*np.pi))*tau_D*a*kT/((c**2)*(sig**2)*l_D))

print('The ratio of shot noise to JN noise is:', S_shot/S_JN)
print('The resolution per Debye time is:', V0*1e3, 'mV')
print('The crossover number of channels is:', N_s)
