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
    e_m,
    n_herm,
    d_fixed,
    s_example,
    h_log_min,
    h_log_max,
    n_points,
    FIGURES_DIR,
)


# ------------------------------------------------------------------------------
# DEFINE INTEGRAND
# ------------------------------------------------------------------------------
def gamma(x, d):
    return (1.0 + (1 / e_m) * np.tanh(x * d / 2)) ** (-1)


def G(x, q, h, d):
    # Components of B
    B1 = (h**2) / ((q * h) ** 2 - 1.0)
    B2 = ((1.0 + q * h) * (gamma(x, d) / (x + q) - q)) ** (-1)
    B3 = (q**2) - (x**2) - gamma(x, d) * (1 - x * h)

    # Components of C
    C1 = -2 * B1 * (q * (h**2)) / ((q * h) ** 2 - 1.0)
    C2 = -(B2**2) * (
        h * (gamma(x, d) / (x + q) - q) - (1 + q * h) * (gamma(x, d) / ((x + q) ** 2) + 1)
    )
    C3 = 2 * q

    # Derivative of components
    G1 = C1 * (2 * B2 * B3 + (1 - (x * h) ** 2) / h)
    G2 = 2 * B1 * (C2 * B3 + B2 * C3)

    return (G1 + G2) * x / q


def unweighted_integrand(x, h, d):
    q = np.sqrt(1 + x**2)
    G_int = G(x, q, h, d)
    return G_int


# ------------------------------------------------------------------------------
# INTEGRATION FUNCTIONS
# ------------------------------------------------------------------------------
# Define range of h to search
h_vals = np.logspace(h_log_min, h_log_max, n_points)


# -----------------------------
# (B) Gauss–Hermite (recommended if you can factor the Gaussian)
# -----------------------------
def hermite_half_range_integral(F, h, s, d, n_herm=n_herm):
    """
    Uses n-point Gauss–Hermite on (-inf, inf) with weight exp(-t^2)
    to compute: ∫_0^∞ exp(-(s x)^2) F(x,h) dx.

    Change of variable: t = s x  => dx = dt / s
    Integral = (1/s) ∫_0^∞ exp(-t^2) F(t/s, h) dt

    We approximate ∫_0^∞ exp(-t^2) G(t) dt = 0.5 ∫_{-∞}^{∞} exp(-t^2) G(|t|) dt
    and apply Hermite to the even extension G(|t|).
    """
    # Hermite nodes and weights for ∫ e^{-t^2} g(t) dt
    x, w = np.polynomial.hermite.hermgauss(n_herm)
    # Even extension via |x| to cover only t >= 0
    G = F(np.abs(x) / s, h, d)
    return (1.0 / s) * 0.5 * np.sum(w * G)


def noise_scaling(s, d):
    # Get raw noise
    S_vals = np.array(
        [hermite_half_range_integral(unweighted_integrand, h, s, d) for h in h_vals]
    )
    S_0 = hermite_half_range_integral(unweighted_integrand, 1.0, s, d)
    # Get SNR values
    chi_vals = h_vals / (1 + h_vals)
    SNR_vals = (chi_vals**2) / S_vals
    SNR_vals /= 1.0 / (4 * S_0)
    # Get optimal height
    idx_opt = np.argmax(SNR_vals)
    h_opt = h_vals[idx_opt]
    SNR_opt = SNR_vals[idx_opt]
    return S_vals, SNR_vals, h_opt, SNR_opt


# ------------------------------------------------------------------------------
# OPTIMIZING OVER SIGNAL-TO-NOISE RATIO
# ------------------------------------------------------------------------------
d = d_fixed  # membrane thickness (units of l_D)
s_vals = s_example


# # FIGURE: SCALING OF NOISE WITH HEIGHT
# fig, ax = plt.subplots(figsize=(.9*width, .9*height))

# ax.plot(h_vals, 8*S_vals*(s**2), lw=2, c='b')

# ax.set_xscale('log')
# ax.set_yscale('log')

# ax.set_ylabel('Noise $S^{M}(\omega)/S^0_{\\text{shot}}(\omega)$')
# ax.set_xlabel('Sensor height $h/l_D$')
# plt.savefig('../figures/noise_height_scaling.pdf', dpi=600, bbox_inches='tight')
# plt.show()

fig, ax = plt.subplots(figsize=(0.9 * width, 0.9 * height))

for i, s in enumerate(s_vals):
    # Get optimal values
    S_vals, SNR_vals, h_opt, SNR_opt = noise_scaling(s, d)

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
plt.show()


# ------------------------------------------------------------------------------
# MODE STRUCTURE
# ------------------------------------------------------------------------------
# FIGURE: SHORT AND LONG WAVELENGTH MODES
h_ex_vals = np.linspace(0.0, 5.0, 100)

fig, ax = plt.subplots(figsize=(0.9 * width, 0.9 * height))

ax.plot([0.0, 5.0], [0.0, 0.0], c="k", lw=1, ls="--")
ax.plot(h_ex_vals, 0.2 * np.cos(5 * h_ex_vals), lw=2, c="r")
ax.plot(h_ex_vals, np.exp(-h_ex_vals), lw=2, c="b")
ax.set_ylim(-0.25, 1.05)

ax.set_ylabel("Charge density mode $\\rho({\\bf k},z)$")
ax.set_xlabel("Distance from membrane $z/l_D$")
plt.show()
