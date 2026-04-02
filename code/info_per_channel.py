#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 14:41:13 2024

@author: jbet
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update(mpl.rcParamsDefault)

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 20,
    "font.serif": ["Garamond"],
})

# Parameters
kB = 1.38e-23
T = 300
alpha = 1
sig = 5e-9
R0 = 1e-6
Ic = 1e-11
tau = 1e-3

gamma = 16*(np.pi**(3/2))*alpha*kB*T*(R0**2)/sig
N_s = 2*np.sqrt(np.pi)*R0/sig
N_0 = gamma/(N_s*tau*(Ic**2))

# N values
n_points = 100
log_N_vals = np.linspace(0,7,n_points)
N_vals = 10**log_N_vals

# Info rate functions
def info_rate(N):
    return np.log(1+((N-1)**2)/(1+(N-1)/N_s)*(Ic**2)*tau/gamma)/np.log(2)


# Plot one-to-many
fig = plt.figure(figsize=(6.4, 4.8*(1.6/2)))
ax = fig.add_subplot(111)
plt.plot(N_vals, info_rate(N_vals), c='b', lw=2)
plt.plot(N_vals, np.log(N_vals/N_0)/np.log(2), c='k', ls='--', lw=1)
plt.plot([N_s,N_s], [0,info_rate(N_s)], c='r', ls='--', lw=1)
plt.xscale('log')
plt.xlabel('Number of channels ($N$)')
plt.ylabel('$\dot{M} \\tau$ (bits)')
plt.show()
