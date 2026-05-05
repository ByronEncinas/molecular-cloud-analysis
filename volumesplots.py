import matplotlib
matplotlib.use("Agg") 
import csv, glob, os, sys, time, h5py
import matplotlib.pyplot as plt
from tabulate import tabulate
import numpy as np
from copy import deepcopy
from scipy.spatial import cKDTree
import warnings
from src.library import *
import pandas as pd
import matplotlib as mpl

mpl.rcParams['text.usetex'] = True
MARKERS = ['v', 'o']
COLORS = [
    "#8E2BAF",  # Deep Purple     — original
    "#148A02",  # Forest Green    — original
    "#C42B8E",  # Magenta Rose    — analogous to purple, warm bridge
    "#AF2B2B",  # Crimson         — bold warm anchor
    "#D4820A",  # Amber           — energetic warm accent
    "#8AAF0A",  # Chartreuse      — yellow-green, bridges to green
    "#0A8A5E",  # Emerald Teal    — cool-green bridge
    "#0A5EAF",  # Royal Blue      — cool counterweight
]
ALPHA   = 0.9
SIZE    = 8
FONTSIZE = 18
GRID_ALPHA = 0.5

start_time = time.time()

data = [
    {
        "snapshot": 100,
        "total_cells": 38,
        "frac_cells_1e2":  1.0,
        "frac_cells_1e4":  0.0,
        "frac_cells_1e6":  0.0,
        "frac_cells_1e8":  0.0,
        "frac_cells_1e10": 0.0,
        "frac_cells_1e12": 0.0,
        "frac_cells_1e14": 0.0,
        "frac_vol_1e2":    1.0,
        "frac_vol_1e4":    0.0,
        "frac_vol_1e6":    0.0,
        "frac_vol_1e8":    0.0,
        "frac_vol_1e10":   0.0,
        "frac_vol_1e12":   0.0,
        "frac_vol_1e14":   0.0,
    },
    {
        "snapshot": 300,
        "total_cells": 259,
        "frac_cells_1e2":  1.0,
        "frac_cells_1e4":  0.06563706563706563,
        "frac_cells_1e6":  0.0,
        "frac_cells_1e8":  0.0,
        "frac_cells_1e10": 0.0,
        "frac_cells_1e12": 0.0,
        "frac_cells_1e14": 0.0,
        "frac_vol_1e2":    1.0,
        "frac_vol_1e4":    0.0031883447714266145,
        "frac_vol_1e6":    0.0,
        "frac_vol_1e8":    0.0,
        "frac_vol_1e10":   0.0,
        "frac_vol_1e12":   0.0,
        "frac_vol_1e14":   0.0,
    },
    {
        "snapshot": 495,
        "total_cells": 685408,
        "frac_cells_1e2":  1.0,
        "frac_cells_1e4":  0.9992311148979878,
        "frac_cells_1e6":  0.8374165460572389,
        "frac_cells_1e8":  0.3315251645735095,
        "frac_cells_1e10": 0.09963992249871609,
        "frac_cells_1e12": 0.044290408048928524,
        "frac_cells_1e14": 0.0,
        "frac_vol_1e2":    1.0,
        "frac_vol_1e4":    0.7849889467512857,
        "frac_vol_1e6":    0.0028566485213035253,
        "frac_vol_1e8":    3.943642662778482e-06,
        "frac_vol_1e10":   5.7045685221958484e-09,
        "frac_vol_1e12":   1.81267005860523e-11,
        "frac_vol_1e14":   0.0,
    },
]

dens = [2,4,6,8,10,12,14]

fig, ax = plt.subplots()

for shot in data:
    vols = []
    cell = []
    _label = shot["snapshot"]
    _title = "+ ambipolar diffusion"
    for l in dens:
        vols += [np.log10(shot[f"frac_vol_1e{l}"])]
        #cell += [shot[f"frac_cells_1e{l}"]]

    ax.set_xlabel(r"$\log_{10}(n_g / \rm{cm}^{-3})$", fontsize=FONTSIZE)
    ax.set_ylabel(r"$\log_{10}(V(n_g;n_c>n_g)/V_{sphere})$ ", fontsize=FONTSIZE)

    ax.set_title(_title, fontsize=FONTSIZE)
    ax.grid(True, which='both', alpha=ALPHA)

    ax.plot(dens, vols,".-", linewidth=2, label=_label)
    fig.tight_layout()
plt.savefig("./series/" + f'avolcompare.png', dpi=150, bbox_inches='tight')
plt.close(fig)

data = [
    {
        "snapshot": "000",
        "total_cells": 1,
        "frac_vol_1e2":    1.0,
        "frac_vol_1e4":    0.0,
        "frac_vol_1e6":    0.0,
        "frac_vol_1e8":    0.0,
        "frac_vol_1e10":   0.0,
        "frac_vol_1e12":   0.0,
        "frac_vol_1e14":   0.0,
    },
    {
        "snapshot": "430",
        "total_cells": 1268355,
        "frac_vol_1e2":    1.0,
        "frac_vol_1e4":    0.7843339610465719,
        "frac_vol_1e6":    0.008461858633394246,
        "frac_vol_1e8":    5.9648383133770995e-06,
        "frac_vol_1e10":   9.793554425240414e-09,
        "frac_vol_1e12":   0.0,
        "frac_vol_1e14":   0.0,
    },
    {
        "snapshot": "495",
        "total_cells": 1328845,
        "frac_vol_1e2":    1.0,
        "frac_vol_1e4":    0.7833321952662746,
        "frac_vol_1e6":    0.008469578685917472,
        "frac_vol_1e8":    6.142886161079583e-06,
        "frac_vol_1e10":   1.0151593836160275e-08,
        "frac_vol_1e12":   2.0021243037267856e-11,
        "frac_vol_1e14":   0.0,
    },
]

fig, ax = plt.subplots()

for shot in data:
    vols = []
    cell = []
    _label = shot["snapshot"]
    _title = "Ideal MHD"
    for l in dens:
        vols += [np.log10(shot[f"frac_vol_1e{l}"])]
        #cell += [shot[f"frac_cells_1e{l}"]]

    ax.set_xlabel(r"$\log_{10}(n_g / \rm{cm}^{-3})$", fontsize=FONTSIZE)
    ax.set_ylabel(r"$\log_{10}(V(n_g;n_c>n_g)/V_{sphere})$ ", fontsize=FONTSIZE)
    ax.set_title(_title, fontsize=FONTSIZE)
    #ax.set_ylim(-0.1, 1.1)
    #ax.set_yscale("log")
    ax.grid(True, which='both', alpha=ALPHA)

    ax.plot(dens, vols,".-", linewidth=2, label=_label)
    fig.tight_layout()
plt.savefig("./series/" + f'ivolcompare.png', dpi=150, bbox_inches='tight')
plt.close(fig)
