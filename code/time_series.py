"""
    This script plots:
        1. An ion channel opening/closing time series (telegraph process)
        2. A voltage trace depicting a few action potentials
-------------------------------------------------------------------------------
created on:
    Wed 3 Dec 2025
-------------------------------------------------------------------------------
last change:
    Tue 11 Feb 2026
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

# ---------- Style (matching snr_scaling.py) -----------------------------------
plt.rcParams.update({
    'text.usetex': False,
    'mathtext.fontset': 'cm',
    'mathtext.rm': 'Helvetica',
    'mathtext.it': 'Helvetica:italic',
    'mathtext.bf': 'Helvetica:bold',
    'font.family': 'Helvetica',
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'axes.linewidth': 0.5,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

mpl.rcParams['figure.dpi'] = 300

logging.getLogger("fontTools").setLevel(logging.ERROR)
logging.getLogger("matplotlib.backends.backend_pdf").setLevel(logging.ERROR)

# Nature figure size
width = 3.5
height = 3.2

con_colors = np.array([[0, 77, 128], [181, 23, 0], [1, 113, 0], [242, 112, 0],
                        [120, 0, 150], [0, 168, 157], [203, 41, 123],
                        [0, 0, 0]]) / 255

np.random.seed(742)

# =============================================================================
# FIGURE 1 – Ion channel opening / closing (telegraph process)
# =============================================================================
dt_ch = 0.01          # time step (ms)
T_ch = 100.0          # total duration (ms)
t_ch = np.arange(0, T_ch, dt_ch)
N_ch = len(t_ch)

# Transition rates (ms^-1)
k_open = 0.05         # closed -> open rate
k_close = 0.3        # open -> closed rate

# Simulate two-state Markov chain
state = np.zeros(N_ch, dtype=int)
state[0] = 0          # start closed
for i in range(1, N_ch):
    if state[i - 1] == 0:                       # currently closed
        state[i] = 1 if np.random.rand() < k_open * dt_ch else 0
    else:                                        # currently open
        state[i] = 0 if np.random.rand() < k_close * dt_ch else 1

# =============================================================================
# FIGURE 2 – Voltage trace with action potentials (Izhikevich model)
# =============================================================================
# Regular-spiking neuron parameters
a_iz = 0.02
b_iz = 0.2
c_iz = -65.0        # reset voltage (mV)
d_iz = 8.0           # reset recovery variable

dt_v = 0.05          # time step (ms)
T_v = 250.0          # total duration (ms)
t_v = np.arange(0, T_v, dt_v)
N_v = len(t_v)

v = np.full(N_v, c_iz)        # membrane potential (mV)
u = np.full(N_v, b_iz * c_iz) # recovery variable

# Constant drive + small noise
I_drive = 10.0
I_ext = I_drive + 1.0 * np.random.randn(N_v)

for i in range(1, N_v):
    if v[i - 1] >= 30.0:
        v[i] = c_iz
        u[i] = u[i - 1] + d_iz
    else:
        dv = (0.04 * v[i - 1] ** 2 + 5 * v[i - 1] + 140
              - u[i - 1] + I_ext[i])
        du = a_iz * (b_iz * v[i - 1] - u[i - 1])
        v[i] = v[i - 1] + dv * dt_v
        u[i] = u[i - 1] + du * dt_v
        if v[i] >= 30.0:
            v[i] = 30.0   # cap at spike peak for plotting

# =============================================================================
# Combined figure – both panels
# =============================================================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(0.9 * width, 0.9 * height))

# Panel A – Ion channel
ax1.step(t_ch, state, where='post', c=con_colors[0], lw=1)
ax1.set_xlabel('Time')
ax1.set_ylabel('Channel current')
ax1.set_yticks([0, 1])
ax1.set_yticklabels(['$0$', '$I_0$'])
ax1.set_xticks([])
ax1.set_ylim(-0.12, 1.12)
ax1.set_xlim(0, T_ch)

# Panel B – Action potentials
ax2.plot(t_v, v, c=con_colors[1], lw=1)
ax2.set_xlabel('Time (ms)')
ax2.set_ylabel('Voltage (mV)')
ax2.set_xlim(0, T_v)

fig.tight_layout()
plt.savefig('../figures/time_series.png', dpi=600, bbox_inches='tight')
plt.show()
