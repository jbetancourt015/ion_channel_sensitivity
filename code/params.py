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

# figures/ is organised one sub-folder per manuscript figure (see the PRL
# submission).  SI figures carry an ``si_`` prefix.  Raw script output goes in a
# ``raw/`` sub-folder; hand-assembled/edited figures sit at the folder root.
#
#   noise_numerics/          PRL Fig 1  (raw: shot_noise_sweep, JN_noise_sweep)
#   spherical_neuron/         PRL Fig 2  (schematic, no raw)
#   snr_saturation/           PRL Fig 3  (raw: snr_saturation)
#   si_communication_scheme/  PRL Fig S1 (schematic, no raw)
#   si_optimal_height/        PRL Fig S2 (raw: snr_height_scaling, optimal_height_sweep)
#   si_noise_param_sweep/     PRL Fig S3 (reuses noise_numerics/raw/*)
#   si_noise_param_sweep_restricted/, si_relative_snr/   not in the manuscript


def figure_path(figure, filename, raw=True):
    """Return an output path inside ``figures/<figure>/`` (or ``.../raw/``).

    ``figure`` is the manuscript-figure folder name, e.g. ``"noise_numerics"``
    or ``"si_optimal_height"``.  The directory is created if needed.
    """
    parts = [FIGURES_DIR, figure] + (["raw"] if raw else [])
    out_dir = os.path.join(*parts)
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, filename)


# ── Physical constants ────────────────────────────────────────────────────────
kT = 4.11e-21  # thermal energy at ~300 K (J)
eps_0 = 8.85e-12  # vacuum permittivity (F/m)

# ── Experimental parameters ─────────────────────────────────────────────────────
alph = 1.0  # ionic conductivity
eps_r = 80.0  # relative permittivity of water
eps = eps_r * eps_0  # water permittivity (F/m)
d = 5e-9  # membrane thickness (m)
c = 1e-2  # membrane capacitance per area (F/m²)

# ── Medium / electrostatic parameters ────────────────────────────────────────
a = 5e-9  # ion channel width (m)
a_lo, a_hi = 1e-9, 5e-9  # a range for bounds
l_D = 8e-10  # Debye screening length (m)
eps_m = d / (1.0 / c - 2 * l_D / eps)  # effective membrane permittivity (F/m)
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
d_norm = d / l_D  # membrane thickness for h-sweep (units of l_D)
s_example = [1, 10, 100]  # a/l_D values to compare in h-sweep plot

# ── Sweep ranges (log10) — param_sweep.py ────────────────────────────────────
s_log_min, s_log_max = 0.0, 2.0  # a / l_D
d_log_min, d_log_max = -1.0 + np.log10(2), 2.0  # d / l_D
h2d_log_min, h2d_log_max = -2.0, 2.5  # h / l_D (height sweep within 2D grid)

# ── Restricted sweep ranges — param_sweep.py (a/l_D and d/l_D from 1 to 10) ─
s_log_min_r, s_log_max_r = 0.0, 2.0  # a / l_D
d_log_min_r, d_log_max_r = 0.0, 1.0  # d / l_D

# ── Numerical results cache ───────────────────────────────────────────────────
NUMERICAL_RESULTS_DIR = os.path.normpath(os.path.join(_here, "..", "numerical_results"))


def results_path(filename):
    """Return a path inside ``repo/numerical_results/`` (created if needed)."""
    os.makedirs(NUMERICAL_RESULTS_DIR, exist_ok=True)
    return os.path.join(NUMERICAL_RESULTS_DIR, filename)
