"""
    This script implements the Blahut-Arimoto algorithm for the navigation
    problem.
-------------------------------------------------------------------------------
created on:
    Thu 23 Oct 2025
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
from matplotlib.colors import LogNorm
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numba import njit
from scipy.linalg import solve_banded
from time import time
from tqdm import tqdm
from noise_integral import noise_scaling

plt.rcParams.update(
    {
        "text.usetex": False,  # keep LaTeX off globally
        "mathtext.fontset": "cm",  # or 'stixsans' for sans-serif
        "mathtext.rm": "Helvetica",
        "mathtext.it": "Helvetica:italic",
        "mathtext.bf": "Helvetica:bold",
        "font.family": "Helvetica",
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.5,  # thin axis lines for publication
        "pdf.fonttype": 42,  # ensures text stays as text in Illustrator
        "ps.fonttype": 42,  # same for PostScript
    }
)

mpl.rcParams["figure.dpi"] = 300

logging.getLogger("fontTools").setLevel(logging.ERROR)
logging.getLogger("matplotlib.backends.backend_pdf").setLevel(logging.ERROR)

# Nature figure size
width = 2.5
height = 2.3

con_colors = (
    np.array(
        [
            [0, 77, 128],
            [181, 23, 0],
            [1, 113, 0],
            [242, 112, 0],
            [120, 0, 150],
            [0, 168, 157],
            [203, 41, 123],
            [0, 0, 0],
        ]
    )
    / 255
)
latex_gray = np.array([240, 240, 240]) / 255

colormap = "cool_r"

# Model parameters
import os
from params import (
    d_norm,
    s_example,
    FIGURES_DIR,
    h_vals,
)

# ------------------------------------------------------------------------------
# OPTIMIZING OVER SIGNAL-TO-NOISE RATIO
# ------------------------------------------------------------------------------
s_vals = s_example

# FIGURE: SCALING OF NOISE WITH HEIGHT
fig, ax = plt.subplots(figsize=(0.9 * width, 0.9 * height))

for i, s in enumerate(s_vals):
    # Get optimal values
    S_vals, SNR_vals, h_opt, SNR_opt = noise_scaling(s, d_norm)

    # FIGURE: SCALING OF SIGNAL-TO-NOISE RATIO WITH HEIGHT
    chi_vals = h_vals / (1 + h_vals)

    ax.plot(h_vals, SNR_vals, lw=2, label=f"$\sigma/l_D={s}$", c=con_colors[i])
    ax.plot([h_opt, h_opt], [0.0, SNR_opt], lw=1, ls="--", c=con_colors[i], alpha=0.5)

ax.legend()
ax.set_xscale("log")
# ax.set_yscale('log')

ax.set_ylabel("$\\text{SNR}(\omega,h)/\\text{SNR}(\omega,l_D)$")
ax.set_xlabel("Sensor height $h/l_D$")
plt.savefig(
    os.path.join(FIGURES_DIR, "snr_height_scaling.svg"), dpi=600, bbox_inches="tight"
)
plt.savefig(
    os.path.join(FIGURES_DIR, "snr_height_scaling.pdf"), dpi=600, bbox_inches="tight"
)
plt.show()

# ------------------------------------------------------------------------------
# POSTER FIGURE
# ------------------------------------------------------------------------------
s_vals = s_example

# FIGURE: SCALING OF NOISE WITH HEIGHT
fig, ax = plt.subplots(figsize=(1.2 * width, 1.2 * height))

for i, s in enumerate(s_vals):
    # Get optimal values
    S_vals, SNR_vals, h_opt, SNR_opt = noise_scaling(s, d_norm)

    # FIGURE: SCALING OF SIGNAL-TO-NOISE RATIO WITH HEIGHT
    chi_vals = h_vals / (1 + h_vals)

    ax.plot(h_vals, SNR_vals, lw=2, label=f"$\sigma/l_D={s}$", c=con_colors[i])

ax.legend()
ax.set_xscale("log")

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.set_ylabel("$\\text{SNR}(\omega,h)/\\text{SNR}(\omega,l_D)$")
ax.set_xlabel("Sensor height $h/l_D$")
plt.savefig(
    os.path.join(FIGURES_DIR, "poster/snr_height_scaling.pdf"), dpi=600, bbox_inches="tight"
)
plt.show()