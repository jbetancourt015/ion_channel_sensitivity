"""
Global parameters for repo/code scripts.
Edit values here and re-run any script to pick up the changes.
-------------------------------------------------------------------------------
"""

import os
import numpy as np

# ── Output paths ──────────────────────────────────────────────────────────────
# Absolute path to figures/, resolved relative to this file so scripts work
# regardless of the working directory they are launched from.
_here = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.normpath(os.path.join(_here, "..", "..", "figures"))

# ── Physical constants ────────────────────────────────────────────────────────
kT = 4.11e-21  # thermal energy at ~300 K (J)
eps_0 = 8.85e-12  # vacuum permittivity (F/m)

# ── Experimental parameters ─────────────────────────────────────────────────────
a = 1.0  # ionic conductivity
eps_r = 80.0  # relative permittivity of water
eps = eps_r * eps_0  # water permittivity (F/m)
d = 5e-9  # membrane thickness (m)
c = 1e-2  # membrane capacitance per area (F/m²)

# ── Medium / electrostatic parameters ────────────────────────────────────────
sig = 5e-9  # ion channel width (m)
sig_lo, sig_hi = 1e-9, 5e-9  # σ range for bounds
l_D = 7e-10  # Debye screening length (m)
eps_m = d / (1.0 / c - 2 * l_D / eps)  # effective membrane permittivity (F/m)
# e_m = 1 / 20.0  # membrane-to-water permittivity ratio ε_m/ε_w
e_m = eps_m / eps  # membrane-to-water permittivity ratio ε_m/ε_w

# ── Soma / cell parameters ────────────────────────────────────────────────────
tau = 1e-3  # membrane time constant (s)
R = 1e-5  # soma radius (m)

# ── Numerical parameters ──────────────────────────────────────────────────────
n_herm = 80  # Gauss–Hermite quadrature order
n_points = 100  # resolution for 1D sweeps
N_params = 100  # grid resolution for 2D parameter sweeps

# ── Sweep ranges (log10) — noise_integral.py ──────────────────────────────────
h_log_min, h_log_max = -2.0, 5.0  # sensor height h / l_D
h_vals = np.logspace(h_log_min, h_log_max, n_points)

# ── Sweep ranges (log10) — snr_scaling.py ────────────────────────────────────
N_log_min, N_log_max = 0.0, 5.0  # channel number N

# ── Sweep ranges (log10) — optimal_height.py ─────────────────────────────────
h_log_min, h_log_max = -2.0, 5.0  # sensor height h / l_D
d_norm = d/l_D  # membrane thickness for h-sweep (units of l_D)
s_example = [1, 10, 100]  # σ/l_D values to compare in h-sweep plot

# ── Sweep ranges (log10) — param_sweep.py ────────────────────────────────────
s_log_min, s_log_max = 0.0, 2.0  # σ / l_D
d_log_min, d_log_max = -1.0 + np.log10(2), 2.0  # d / l_D
h2d_log_min, h2d_log_max = -2.0, 2.5  # h / l_D (height sweep within 2D grid)
