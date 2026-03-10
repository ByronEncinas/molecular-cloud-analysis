import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib as mpl
from src.library import *
from scipy.stats import skew
from scipy.stats import kurtosis
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import os, glob

mpl.rcParams['text.usetex'] = True
# Constants for consistent style
MARKERS = ['v', 'o']
COLORS  = ["#8E2BAF", "#148A02"]
ALPHA   = 0.9
SIZE    = 8
FONTSIZE = 12
GRID_ALPHA = 0.5

def imporfromfile(file, identifier):

    df = pd.read_pickle(file)
    df.index.name = 'snapshot'
    df.index = df.index.astype(int)
    df = df.sort_index()
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)

    t     = df["time"].to_numpy()
    _ = np.argsort(t)

    t     = df["time"].to_numpy()[_]
    x     = df["x_input"].to_numpy()[_]
    n     = df["n_rs"].to_numpy()[_]
    B     = df["B_rs"].to_numpy()[_]
    Nlos0 = df["n_los0"].to_numpy() [_] # mean
    Nlos1 = df["n_los1"].to_numpy() [_] # median
    Ncrs  = df["n_path"].to_numpy()[_]
    factu = df["r_u"].to_numpy()[_]
    factl = df["r_l"].to_numpy()[_]
    surf  = df["surv_fraction"].to_numpy()[_]
    rurl = [np.mean(ru-rl) for (ru,rl) in zip(factu, factl)]

    if identifier == '6i4'   or identifier == '6a4':
        radius = np.max([np.max(ris) for ris in x])
    elif identifier == '4i3' or identifier == '4a3':
        radius = 0.1
    elif identifier == '2i2' or identifier == '2a2':
        radius = 0.5
    elif identifier == '2i1' or identifier == '2a1':
        radius = 0.2
    elif identifier == '2i0' or identifier == '2a0':
        radius = 0.1

    return {
        "id": identifier,
        "rloc": radius,
        "t":     t,
        "x":     x,
        "n":     n,
        "B":     B,
        "Nlos0": Nlos0,
        "Nlos1": Nlos1,
        "Ncrs":  Ncrs,
        "factu": factu,
        "factl": factl,
        "surf":  surf,
        "rurl":  rurl
    }

files = glob.glob(f'./series/data_*i*.pkl')

assert len(files) == 5, "Some IC is missing"

if files:    
    config = {}
    for file in files:
        ID = file.split('.')[-2][-3:]
        print(file, ID)
        os.makedirs(f"./series/{ID}", exist_ok=True)
        config[f'data{ID}'] = imporfromfile(file,ID)

    globals().update(config) # This injects every key as a variable in your script

print(data2i0.keys())
print(data2i1.keys())
print(data2i2.keys())
print(data4i3.keys())
print(data6i4.keys())

# Plotting individual distributions

for __, data in enumerate([data2i0, data2i1, data2i2, data4i3, data6i4]):
    directory = "./series/" + data["id"] + '/'
    print("\nOutput will go into: ", directory)

    #if __ == 0:

    
    if 'i' in directory:
        x1, x2 = 4.5515, 4.5521
    else:
        x1, x2 = 4.2904, 4.2905

    max_radius = data['rloc']
    ratio0 = np.array([np.mean(nlos0/ncrs) for (nlos0,ncrs) in zip(data["Nlos0"],data["Ncrs"])]) 
    ratio1 = np.array([np.median(nlos1/ncrs) for (nlos1,ncrs) in zip(data["Nlos0"],data["Ncrs"])])
    ratio2 = np.array([np.std(nlos0/ncrs) for (nlos0,ncrs) in zip(data["Nlos0"],data["Ncrs"])])

    fig, axd = plt.subplot_mosaic(
        [["left", "right"]],
        sharey=True,
        gridspec_kw={"width_ratios": [3, 1], "wspace": 0.05}
    )

    title = r"Mean ratio $N_{los}/N_{crs}$"

    axd["left"].plot(data['t'], ratio0, '.-')
    axd["left"].set_ylabel("$N_{los}/N_{crs}$",fontsize=FONTSIZE)
    axd["left"].set_xlabel("Time [Myrs]",fontsize=FONTSIZE)
    axd["left"].grid(True, which='both', alpha=0.5)
    axd["left"].set_ylim(0.0, 2.0)
    axd["right"].plot((data['t']-x1)*1_000, ratio0, ".-")
    axd["right"].set_xlabel("Time [kyrs]",fontsize=FONTSIZE)
    axd["right"].set_xlim((x1-x1)*1_000,(x2-x1)*1_000)
    axd["right"].tick_params(labelleft=False)
    axd["right"].grid(True, which='both', alpha=0.5)
    axd["right"].set_ylim(0.0, 2.0)
    fig.suptitle(title)
    plt.savefig(directory + 'ratio0.png', dpi=300)
    #plt.show()
    plt.close()

    fig, axd = plt.subplot_mosaic(
        [["left", "right"]],
        sharey=True,
        gridspec_kw={"width_ratios": [3, 1], "wspace": 0.05}
    )

    title = r"Median ratio $N_{los}/N_{crs}$"

    axd["left"].plot(data['t'], ratio1, '.-')
    axd["left"].set_ylabel("$N_{los}/N_{crs}$",fontsize=FONTSIZE)
    axd["left"].set_xlabel("Time [Myrs]",fontsize=FONTSIZE)
    axd["left"].grid(True, which='both', alpha=0.5)
    axd["left"].set_ylim(0.0, 2.0)

    axd["right"].plot((data['t']-x1)*1_000, ratio1, ".-")
    axd["right"].set_xlabel("Time [kyrs]",fontsize=FONTSIZE)
    axd["right"].set_xlim((x1-x1)*1_000,(x2-x1)*1_000)
    axd["right"].tick_params(labelleft=False)
    axd["right"].grid(True, which='both', alpha=0.5)
    axd["right"].set_ylim(0.0, 2.0)
    fig.suptitle(title)
    plt.savefig(directory + 'ratio1.png', dpi=300)
    #plt.show()
    plt.close()

    
    """
    Interpretation:
    Positive skew: The majority of data points are closer to the minimum.
    Negative skew: The majority of data points are closer to the maximum.
    Skewness of 0: The data is perfectly symmetric around the mean

    Interpretation of Fisher (Excess) Kurtosis:

    Excess kurtosis > 0: The distribution has heavier tails than a normal distribution.
    Excess kurtosis < 0: The distribution has lighter tails than a normal distribution.
    Excess kurtosis = 0: The distribution has the same tail behavior as a normal distribution.
    """


    f_mean   = np.array([np.mean(ul[ul<1]) for ul in data['factu']])
    f_median = np.array([np.median(ul[ul<1]) for ul in data['factu']])
    f_std    = np.array([np.std(ul[ul<1]) for ul in data['factu']])
    f_skew   = np.array([skew(ul[ul<1]) for ul in data['factu']])
    f_kurt   = np.array([kurtosis(ul[ul<1]) for ul in data['factu']])
    f_less   = np.array([np.sum(ul<1)/ul.shape[0] for ul in data['factu']])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.set_xlabel(r"Time [Myrs]", fontsize=FONTSIZE)
    ax1.set_ylabel(r"$n_{local}/n_{ism}$ [Adim]", fontsize=FONTSIZE)
    ax1.grid(True, which='both', alpha=GRID_ALPHA)
    ax1.plot(data['t'], f_mean, '-', color='black', alpha=ALPHA, label=r'Mean')
    ax1.plot(data['t'], f_median, '--', color='black', alpha=ALPHA, label=r'Median')
    ax1.plot(data['t'], f_less, '.-', color='orange', alpha=ALPHA, label=r'$f_{R<1}$')
    ax1.legend(fontsize=FONTSIZE-2)

    percentiles = [0, 10, 20, 30, 40]  #percentiles = np.linspace(0,37.5,4) 

    for ptile in percentiles[::-1]:  # plot largest band first for proper layering
        f_ptile_down = np.array([np.percentile(ul[ul<1], ptile) for ul in data["factu"]])
        f_ptile_up   = np.array([np.percentile(ul[ul<1], 100-ptile) for ul in data["factu"]])
        
        ax1.fill_between(data['t'], f_ptile_down, f_ptile_up,
                        color=COLORS[0], alpha=0.3, label=f'{ptile}–{100-ptile} percentile', zorder=1)

        ax1.text(data['t'][4], f_ptile_up[4], f"P{100-ptile}",
            fontsize=6,
            color="black",
            alpha=1.0,
            rotation=0,
            ha="left",   
            va="bottom") 

        ax1.text(data['t'][4], f_ptile_down[4], f"P{ptile}",
            fontsize=6,
            color="black",
            alpha=1.0,
            rotation=0,
            ha="left",   # horizontal alignment: left, center, right
            va="bottom") # vertical alignment: top, center, bottom
        
    ax1.set_ylim(-0.1,1.1)
    ax2.set_xlabel(r"Time [Myrs]", fontsize=FONTSIZE)
    ax2.grid(True, which='both', alpha=GRID_ALPHA)
    ax2.plot(data['t'], f_skew, '-', color='black', alpha=ALPHA, label=r'$\gamma$ : Skewness')
    ax2.plot(data['t'], f_kurt, '--', color='black', alpha=ALPHA, label=r'$\kappa$: Kurtosis')
    ax2.legend(fontsize=FONTSIZE-2)

    axins = inset_axes(ax2, width="40%", height="40%", loc='upper right')  # adjust size and location
    axins.set_xlabel(r"Time [kyrs]", fontsize=FONTSIZE)
    axins.plot((data['t']-x1)*1_000, f_skew, '-', color='black', alpha=ALPHA)
    axins.plot((data['t']-x1)*1_000, f_kurt, '--', color='black', alpha=ALPHA)
    axins.scatter((data['t']-x1)*1_000, f_skew, color='black', alpha=ALPHA, s=SIZE)
    axins.scatter((data['t']-x1)*1_000, f_kurt, color='black', alpha=ALPHA, s=SIZE)

    at_colapse = data['t'] > x1

    y1, y2 = min(np.min(f_skew[at_colapse]), np.min(f_kurt[at_colapse]))*1.5, max(np.max(f_skew[at_colapse]), np.max(f_kurt[at_colapse]))*1.5
    axins.set_xlim((x1-x1)*1_000, (x2-x1)*1_000)
    axins.set_ylim(y1, y2)
    axins.grid(True, which='both', alpha=0.5)

    plt.savefig(directory + 'rfstats.png', dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots()

    n_plots = 1

    ax.set_xlabel(r"$n_g$ [cm$^{-3}$]", fontsize=FONTSIZE)
    ax.set_ylabel(r"$n_{local}/ n_{ism}$ [Adim]", fontsize=FONTSIZE)
    ax.grid(True, which='both', alpha=GRID_ALPHA)
    ax.set_xscale('log')
    ax.set_ylim(-0.1, 1.1)

    cmap = plt.get_cmap("viridis")
    t_min, t_max = min(data["t"][::n_plots]), max(data["t"][::n_plots])

    # Plot lines with colors
    for i, (t_val, rt_val) in enumerate(zip(data["t"][::n_plots], data["factu"][::n_plots])):

        _l_ = rt_val < 1
        #print(rt_val.shape, np.isnan(rt_val))

        rt_val = rt_val[_l_]
        n_val = data["n"][i][_l_]
        n_ref, mean_vec, median_vec, ten_vec, sample_size = red_to_den(rt_val, n_val)
        #n_ref, r_matrix, mean_vec, median_vec, ten_vec, sample_size = reduction_to_density(rt_val, n_val)
        normalized_t = (t_val - t_min) / (t_max - t_min)        
        color = cmap(normalized_t)
        ax.plot(n_ref, mean_vec, '-', lw=2, color=color, alpha=0.6)
        ax.set_xlim(10**2, 10**10)

    norm = mpl.colors.Normalize(vmin=t_min, vmax=t_max)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Time [Myrs]', fontsize=FONTSIZE)

    fig.tight_layout()
    plt.savefig(directory + 'rfdendistro.png', dpi=300)
    plt.close(fig)


# Plotting comparisons between ICs
if __name__ == '__main__':

    """
    REDUCTION FACTOR

    \frac{\mathcal{n}_{local}}{\mathcal{n}_{ism}} = 1 - \sqrt{1-\frac{B(s)}{B_i}}
    
    """
    # plot and save Reduction factor over n_g using log10(n_g(p)/n_{ref}) < 1/8

    output = 'r_n_'
    n_plots = 1

    fig, ax = plt.subplots()

    ax.set_xlabel(r"$n_g$ [cm$^{-3}$]", fontsize=FONTSIZE)
    ax.set_ylabel(r"$1 - n_{local}/ n_{ism}$ [Adim]", fontsize=FONTSIZE)
    ax.grid(True, which='both', alpha=GRID_ALPHA)
    ax.set_xscale('log')
    ax.set_ylim(-0.1, 1.1)

    cmap = plt.get_cmap("viridis")
    t_min, t_max = min(t[::n_plots]), max(t[::n_plots])

    # Plot lines with colors
    for i, (t_val, rt_val) in enumerate(zip(t[::n_plots], factu[::n_plots])):
        _l_ = rt_val < 1
        rt_val = rt_val[_l_]
        n_val = n[i][_l_]
        n_ref, mean_vec, median_vec, ten_vec, sample_size = red_to_den(rt_val, n_val)
        #n_ref, r_matrix, mean_vec, median_vec, ten_vec, sample_size = reduction_to_density(rt_val, n_val)
        normalized_t = (t_val - t_min) / (t_max - t_min)        
        color = cmap(normalized_t)
        ax.plot(n_ref, 1 - mean_vec, '-', lw=2.0, color=color, alpha=1.0)

    norm = mpl.colors.Normalize(vmin=t_min, vmax=t_max)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Time [Myrs]', fontsize=FONTSIZE)

    fig.tight_layout()
    plt.savefig('series/' + output + f'{INPUT}.png', dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots()

    ax.set_xlabel(r"$n_g$ [cm$^{-3}$]", fontsize=FONTSIZE)
    ax.set_ylabel(r"$n_{local}/ n_{ism}$ [Adim]", fontsize=FONTSIZE)
    ax.grid(True, which='both', alpha=GRID_ALPHA)
    ax.set_xscale('log')
    ax.set_ylim(-0.1, 1.1)

    cmap = plt.get_cmap("viridis")
    t_min, t_max = min(t[::n_plots]), max(t[::n_plots])

    # Plot lines with colors
    for i, (t_val, rt_val) in enumerate(zip(t[::n_plots], factu[::n_plots])):

        _l_ = rt_val < 1
        rt_val = rt_val[_l_]
        n_val = n[i][_l_]
        n_ref, r_matrix, mean_vec, median_vec, ten_vec, sample_size = reduction_to_density(rt_val, n_val)
        normalized_t = (t_val - t_min) / (t_max - t_min)        
        color = cmap(normalized_t)
        ax.plot(n_ref, mean_vec, '-', lw=2, color=color, alpha=0.6)

    norm = mpl.colors.Normalize(vmin=t_min, vmax=t_max)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Time [Myrs]', fontsize=FONTSIZE)

    fig.tight_layout()
    plt.savefig('series/' + output + f'2{INPUT}.png', dpi=300)
    plt.close(fig)

    """
    COLUMN DENSITIES

    \int_0^s \frac{n_g(s')ds'}{\hat{\mu}}
    
    """


    output = 'los_crs_'

    fig, ax1 = plt.subplots()

    ax1.set_xlabel(r"Time [Myrs]", fontsize=FONTSIZE)
    ax1.set_ylabel(r"$N_{los}/N_{crs}$ [Adim]", fontsize=FONTSIZE)
    ax1.set_yscale('log')
    ax1.grid(True, which='both', alpha=GRID_ALPHA)

    ax1.plot(t, ratio0, '-', color='black', alpha=ALPHA, label=r'Mean')
    ax1.plot(t, ratio1, '--', color='black', alpha=ALPHA, label=r'Median')

    ax1.legend(fontsize=FONTSIZE-2)

    for ptile in percentiles[::-1]:  # plot largest band first for proper layering
        f_ptile_down = np.array([np.percentile(nlos0/ncrs, ptile) for (nlos0,ncrs) in zip(Nlos0,Ncrs)])
        f_ptile_up   = np.array([np.percentile(nlos0/ncrs, 100-ptile) for (nlos0,ncrs) in zip(Nlos0,Ncrs)])
        
        ax1.fill_between(t, f_ptile_down, f_ptile_up,
                        color=COLORS[0], alpha=0.3, label=f'{ptile}–{100-ptile} percentile', zorder=1)
        """
        ax1.text(t[4], f_ptile_up[4], f"P{100-ptile}",
            fontsize=6,
            color="black",
            alpha=1.0,
            rotation=0,
            ha="left",   # horizontal alignment: left, center, right
            va="bottom") # vertical alignment: top, center, bottom

        ax1.text(t[4], f_ptile_down[4], f"P{ptile}",
            fontsize=6,
            color="black",
            alpha=1.0,
            rotation=0,
            ha="left",   # horizontal alignment: left, center, right
            va="bottom") # vertical alignment: top, center, bottom
        """
    axins = inset_axes(ax1, width="30%", height="30%", loc='lower left')  # adjust size and location
    axins.plot(t, ratio0, '-', color='black', alpha=ALPHA)
    axins.plot(t, ratio1, '--', color='black', alpha=ALPHA)

    x1, x2 = 4.550, 4.5523
    y1, y2 = min(np.min(ratio0), np.min(ratio1))*0.75, max(np.max(ratio0), np.max(ratio1))*1.5
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.grid(True, which='both', alpha=0.5)

    plt.savefig('series/' + output + f'{INPUT}.png', dpi=300)
    plt.close(fig)

