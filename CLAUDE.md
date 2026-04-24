# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research pipeline for a manuscript on ion channel sensitivity and noise in neural membranes. The project computes the signal-to-noise ratio (SNR) for voltage sensing by ion channels, derives the optimal sensor geometry, and characterizes competing noise sources (shot noise vs. Johnson-Nyquist noise). Publication-ready scripts live in `code/`; `../exploratory/` contains intermediate code; `../calculations/` holds Mathematica notebooks and compiled derivation PDFs.

## Environment Setup

```bash
python -m venv venv
source venv/bin/activate
pip install numpy scipy matplotlib numba tqdm
```

Key dependencies: numpy, scipy, matplotlib, numba (JIT for cost functions), tqdm.

## Execution

All `code/` scripts write figures to `../figures/` at the project root. The output path is resolved from `params.py`'s location, so scripts work correctly regardless of working directory. Typical invocation:

```bash
cd code

# Core SNR theory
python snr_scaling.py          # SNR vs. N channels; prints N*, shot/JN ratio, V0

# Noise integrals and optimal geometry
python optimal_height.py       # SNR vs. sensor height h for several σ values

# Parameter sweeps (slow: 100×100 grid with Gauss–Hermite integration)
python param_sweep.py          # heatmaps: optimal h*, S^M/S_shot, S^M/S_JN vs. (σ, d)
```

`param_sweep.py` is the most compute-intensive script (≈ minutes for the 100×100 grid).

## Architecture

### Data Flow

```
code/*.py  (standalone scripts)
    → ../figures/*.pdf / *.svg
```

All scripts share a single parameters file: **`code/params.py`**. Edit values there and re-run any script to pick up the change. Each script imports only what it needs via `from params import ...`.

### Physical Model

All quantities are in SI unless stated otherwise. Recurring symbols across scripts:

| Symbol | Meaning | Typical value |
|--------|---------|---------------|
| `l_D` | Debye screening length | 7×10⁻¹⁰ m |
| `sig` (`σ`) | Ion channel radius | 5×10⁻⁹ m |
| `tau_D` | Debye time `ε/a` | derived |
| `e_m` | Membrane permittivity ratio `ε_m/ε_w` | 1/20 |
| `d` | Membrane thickness (dimensionless, in units of `l_D`) | varied |
| `h` | Sensor height above membrane (in units of `l_D`) | varied |
| `N_s` | Crossover channel number | derived |
| `h_opt` | Optimal sensor height maximizing SNR | derived |

### Key Scripts

- **`snr_scaling.py`** — Plots `SNR_N(ω)/SNR_cell(ω)` vs. N; computes the crossover `N*` at which additional channels stop improving SNR. Prints `N*`, shot/JN noise ratio, and `V0` (resolution per Debye time).

- **`optimal_height.py`** — Implements Gauss–Hermite integration (`hermite_half_range_integral`) of the dimensionless noise kernel `G(x, q, h)` (from SI section S3.5). Sweeps sensor height `h` and plots SNR normalized to `h = l_D`.

- **`param_sweep.py`** — Runs the same noise integral on a 100×100 (σ, d) grid. Produces three heatmaps: optimal height `h*`, noise relative to shot noise floor, and noise relative to JN noise. The `LogMidpointNormalize` class centers diverging colormaps at ratio = 1.

### Figure Style

Consistent across all current `code/` scripts:

```python
plt.rcParams.update({
    'font.family': 'Helvetica',
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'axes.linewidth': 0.5,
    'pdf.fonttype': 42,   # text stays as text in Illustrator
    'ps.fonttype': 42,
})
mpl.rcParams['figure.dpi'] = 300
```

Nature-format figure sizes: `width = 2.5` or `3.5` inches. Figures saved as both PDF and SVG where possible.

Color palette (`con_colors`, 8 colors, RGB/255):
```python
con_colors = np.array([[0,77,128],[181,23,0],[1,113,0],[242,112,0],
                        [120,0,150],[0,168,157],[203,41,123],[0,0,0]])/255
```

### Exploratory / Legacy Code

- `../exploratory/code/` — Scripts moved out of `code/` once superseded:
  - `time_series.py` — Ion channel telegraph process + Izhikevich voltage trace (RNG seed 742)
  - `cost_scaling.py` / `info_per_channel.py` — Cost and information-rate scaling; use old LaTeX/Garamond style
- `../exploratory/bessel_plots.py` — Bessel function plots from early model exploration
- `../figures/exploratory/` — Figures from exploratory phase (not for publication)

### Calculations

- `../calculations/notebooks/` — Mathematica notebooks (`.nb`) for analytical derivations: `info_per_channel`, `mult_channels_plane`, `plate_sensing`, `spherical`, `volume_sensing`
- `../calculations/markdowns/` — Compiled PDFs of theoretical write-ups: bulk fluctuations, bulk measurement, flicking, membraneless geometry, new basis, parametric ansatz, potential fluctuations

### Other Project Directories

- `../submissions/` — Manuscript submission materials
- `../updates/` — Project update notes or incremental results
