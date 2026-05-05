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

cases = ['ideal', 'amb']

# Single Cloud Peak Densities

for case in cases:
    ## this Clouds

    if case == "ideal":
        _label = "ideal MHD"
    else:
        _label = "non-ideal MHD"

    thisclouddf = pd.read_csv(f'./util/{case}_clouds.csv')

    fig, ax = plt.subplots()

    for cloud_name, group in thisclouddf.groupby('cloud'):
        ax.plot(group['time_value'], group['Peak_Density'], "-", linewidth = 2, label=case[0] + cloud_name[-1])

    ax.set_xlabel(r'Elapsed Time [Myrs]', fontsize=FONTSIZE)
    ax.set_ylabel(r'log$_{10}(n_{gas})$ [cm$^{-3}$]', fontsize=FONTSIZE)
    ax.grid(True, which='both', alpha=ALPHA)
    ax.legend(fontsize =FONTSIZE -4)
    #ax.set_title('Cloud Trajectories',fontsize=FONTSIZE)
    plt.yscale("log")
    plt.savefig(f'./series/{case}_thisclouds.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

# Multiple Clouds Peak Densities

fig, ax = plt.subplot_mosaic(
    [[cases[0], cases[1]]],
    sharey=True,
    gridspec_kw={"width_ratios": [1, 1], "wspace": 0.0}
)

ax[cases[0]].set_ylabel(r'log$_{10}(n_{core})$ [cm$^{-3}$]', fontsize=FONTSIZE)

for case in cases:    

    if case == "ideal":
        _label = "ideal MHD"
    else:
        _label = "+ ambipolar diffusion"#"+ non-ideal MHD"

    ax[case].set_title(_label,fontsize=FONTSIZE)

    #ax[case].set_xlabel(r'Time [Myrs]', fontsize=FONTSIZE-4)
    ax[case].set_xlabel(r'Time', fontsize=FONTSIZE-4)
    ax[case].grid(True, which='both', alpha=ALPHA)
    #ax[case].set_yscale("log")  # set per-axis, not via plt.yscale
    ax[case].set_xlim(-0.11, 4.7)

    thiscloud = pd.read_csv(f'./util/{case}_clouds.csv')

    for cloud_name, group in thiscloud.groupby('cloud'):
        ax[case].plot(group['time_value'], np.log10(group['Peak_Density']), "-", linewidth=2,
                      label=case[0] + cloud_name[0] + cloud_name[-1])
    ax[case].legend(fontsize=FONTSIZE - 4, loc = "upper left")

plt.setp(ax[cases[0]].get_xticklabels(), fontsize = FONTSIZE)
plt.setp(ax[cases[1]].get_xticklabels(), fontsize = FONTSIZE)
plt.setp(ax[cases[0]].get_yticklabels(), fontsize = FONTSIZE)
plt.savefig(f'./paper/thisclouds.png', dpi=150, bbox_inches='tight')
plt.close(fig)

# Positions in Space

fig, ax = plt.subplot_mosaic(
    [["u" + cases[0], "u" + cases[1]],
    ["l" + cases[0], "l" + cases[1]]],
    sharey=True, 
    gridspec_kw={"width_ratios": [1, 1], "wspace": 0.2, "hspace": 0.4}
)

for case in cases:    

    if case == "ideal":
        _label = "ideal MHD"
    else:
        _label = "+ ambipolar diffusion"#"+ non-ideal MHD"

    thiscloud = pd.read_csv(f'./util/{case}_clouds.csv')

    _case = "u"+case
    if "ua" not in _case:
        ax[_case].set_ylabel(r'Y [Pc]', fontsize=FONTSIZE-4)
    
    ax[_case].set_xlabel(r'X [Pc]', fontsize=FONTSIZE-4)

    ax[_case].set_ylim(150,250)

    ax[_case].set_title(_label,fontsize=FONTSIZE)

    ax[_case].grid(True, which='both', alpha=ALPHA)

    for cloud_name, group in thiscloud.groupby('cloud'):
        key_init = group['c_coord_Z'].keys()[0]
        key_last = group['c_coord_Z'].keys()[-1]
        ax[_case].scatter(group['c_coord_X'][key_init], group['c_coord_Y'][key_init], marker = MARKERS[0], color="green")
        ax[_case].scatter(group['c_coord_X'][key_last], group['c_coord_Y'][key_last], marker = MARKERS[0], color="red")
        ax[_case].plot(group['c_coord_X'], group['c_coord_Y'], "-", linewidth=2,
                      label=case[0] + cloud_name[0] + cloud_name[-1], color="black", alpha=0.2)

    _case = "l"+case
    if "la" not in _case:
        ax[_case].set_ylabel(r'X [Pc]', fontsize=FONTSIZE-4)

    ax[_case].set_ylim(100,250)
    ax[_case].set_xlim(20,80)

    ax[_case].set_xlabel(r'Z [Pc]', fontsize=FONTSIZE-4)
    #ax[case].set_title(_label,fontsize=FONTSIZE)

    ax[_case].grid(True, which='both', alpha=ALPHA)
    for cloud_name, group in thiscloud.groupby('cloud'):

        key_init = group['c_coord_Z'].keys()[0]
        key_last = group['c_coord_Z'].keys()[-1]

        ax[_case].scatter(group['c_coord_Z'][key_init], group['c_coord_X'][key_init], marker = MARKERS[0], color="green")
        ax[_case].scatter(group['c_coord_Z'][key_last], group['c_coord_X'][key_last], marker = MARKERS[0], color="red")
        ax[_case].plot(group['c_coord_Z'], group['c_coord_X'], "-", linewidth=2,
                      label=case[0] + cloud_name[0] + cloud_name[-1], color="black", alpha=0.2)
    #ax[case].legend(fontsize=FONTSIZE - 8, loc = "lower left", frameon=False)

plt.savefig(f'./paper/binarythisclouds.png', dpi=150, bbox_inches='tight')
plt.close(fig)

# Velocities

fig, ax = plt.subplot_mosaic(
    [[cases[0], cases[1]]],
    sharey=True,
    gridspec_kw={"width_ratios": [1, 1], "wspace": 0.0}
)

ax[cases[0]].set_ylabel(r'Velocity [Pc Myrs$^{-1}$]', fontsize=FONTSIZE)

for case in cases:    

    if case == "ideal":
        _label = "ideal MHD"
    else:
        _label = "+ ambipolar diffusion"#"+ non-ideal MHD"

    thiscloud = pd.read_csv(f'./util/{case}_clouds.csv')

    ax[case].set_title(_label,fontsize=FONTSIZE)

    ax[case].set_xlabel(r'Time [Myrs]', fontsize=FONTSIZE-4)

    ax[case].grid(True, which='both', alpha=ALPHA)

    ax[case].set_xlim(3.0*(1.-0.025), np.max(thiscloud["time_value"]*1.025))

    for cloud_name, group in thiscloud.groupby('cloud'):
        if cloud_name != "cloud-0":
            continue

        group = group.sort_values('time_value')  # ensure time order

        dx = group['c_coord_X'].rolling(5).mean().diff().to_numpy()
        dy = group['c_coord_Y'].rolling(5).mean().diff().to_numpy()
        dz = group['c_coord_Z'].rolling(5).mean().diff().to_numpy()
        dt = group['time_value'].diff().to_numpy()

        vl = np.sqrt(dx*dx + dy*dy + dz*dz) / dt
        t  = group['time_value'].to_numpy()
        absvl =  np.sqrt((dx[-1] - dx[5])**2 + (dy[-1] - dy[5])**2 + (dz[-1] - dz[5])**2) / abs(dt[-1] - dt[5])
        mask = t>3.0
    

        ax[case].plot(t[mask],  vl[mask], "-", linewidth=2,
                      label=case[0] + cloud_name[0] + cloud_name[-1])
        ax[case].plot(t[mask],  absvl*vl[mask]/vl[mask], "--", linewidth=2,
                      label=case[0] + cloud_name[0] + cloud_name[-1])
        #ax[case].plot(t, absvl*np.ones(t.shape), "--", linewidth=2, label="Mean")
        #ax[case].legend(fontsize=FONTSIZE - 4, loc = "upper left")

plt.savefig(f'./paper/thiscloudvel.png', dpi=150, bbox_inches='tight')

plt.close(fig)