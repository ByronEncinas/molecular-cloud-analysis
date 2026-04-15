from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import skew, kurtosis
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use("Agg")  # must be FIRST, before pyplot or anything else
import matplotlib.pyplot as plt
import matplotlib as mpl
from copy import deepcopy
#from src.library import *
import pandas as pd
import numpy as np
import os, glob, sys, time
import numpy as np
import cv2
import glob
mpl.rcParams['text.usetex'] = True
MARKERS = ['v', 'o']
COLORS  = ["#8E2BAF", "#148A02"]
ALPHA   = 0.9
SIZE    = 8
FONTSIZE = 18
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

    radius = np.ceil(np.max([np.max(ris) for ris in x])*100)/100

    if identifier == '6i4'   or identifier == '6a4':
        radius = np.ceil(np.max([np.max(ris) for ris in x])*100)/100
    elif identifier == '4i3' or identifier == '4a3':
        radius = 0.1
    elif identifier == '2i2' or identifier == '2a2':
        radius = 0.5
    elif identifier == '2i1' or identifier == '2a1':
        radius = 0.2
    elif identifier == '2i0' or identifier == '2a0':
        radius = 0.1
        


    if "e-" in identifier:
        radius = float(f"1.0{identifier[1:]}")
    print(radius)

    

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

fig = plt.figure()
plt.ion()

def render(t, x, y, d, identifier ,bins=100):
    fig.clf()
    ax = fig.add_subplot(111)
    ax.set_title(f"{identifier} - t = {t}" )
    divisions = x.shape[0]//10
    print(divisions)

    if d is not None:
        xi = np.linspace(x.min(), x.max(), divisions)
        yi = np.linspace(y.min(), y.max(), divisions)
        xi, yi = np.meshgrid(xi, yi)
        zi = griddata((x, y), d, (xi, yi), method="linear")
        im = ax.imshow(
            zi, origin="lower", aspect="auto",
            extent=[x.min(), x.max(), y.min(), y.max()],
            cmap="inferno"
        )
        fig.colorbar(im, label="density")
    else:
        counts, xedges, yedges = np.histogram2d(x, y, bins=bins)
        smoothed = gaussian_filter(d, sigma=5)
        im = ax.imshow(
            smoothed.T, origin="lower", aspect="auto",
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            cmap="inferno"
        )
        fig.colorbar(im, label="density")

    fig.canvas.draw()
    fig.canvas.flush_events()
    name = str(t).split(".")[0] + str(t).split(".")[1]
    fig.savefig(f"./series/{_id}/{name}.png")
    plt.pause(0.2)

def _animate(_id):
    frames = sorted(glob.glob(f"./series/{_id}/3*.png") + glob.glob(f"./series/{_id}/4*.png"))
    
    first = cv2.imread(frames[0])
    h, w = first.shape[:2]

    out = cv2.VideoWriter(f"./series/{_id}/out.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps=4, frameSize=(w, h))

    for f in frames:
        out.write(cv2.imread(f))

    out.release()

if len(sys.argv) == 1:
    files = glob.glob(f'./series/table/*.pkl')
    print(files)
    n = -3
elif sys.argv[1] == "-e":
    files = glob.glob(f'./series/rloc/*.pkl')
    n = -4
    print(files)

#assert len(files) == 5, "Some IC is missing"

if files:    
    config = {}
    for file in files:
        ID = file.split('.')[-2][n:]
        print(ID, file)
        os.makedirs(f"./series/{ID}", exist_ok=True)
        config[f'data{ID}'] = imporfromfile(file,ID)

    globals().update(config) # This injects every key as a variable in your script


if len(sys.argv) == 1:
    datas = [data6i4, data2i0, data2i1, data2i2, data4i3]
elif sys.argv[1] == "-e":
    datas = []
    for f in files:
        ID = f.split('.')[-2][n:]
        datas += [globals()[f'data{ID}']]

print(len(datas))

""" Re-Make plot 3.1 of the thesis for data 2*0"""

# chosse three moments in time, maybe form the plot of 't'
n_min = np.inf
n_max = 0.0

for _, data_j in enumerate(datas): 
    
    _id     =  data_j["id"]
    total_snaps = data_j['t'].shape[0]
    for j in range(total_snaps): 
        factu   = data_j["factu"][j]
        mask = factu < 1
        try:
            n_min = min(n_min, data_j["n"][j][mask].min())*(1-0.01)
            n_max = max(n_max, data_j["n"][j][mask].max())*1.01
        except:
            continue

    for j in range(total_snaps-1,0,-1):

        _time   = data_j["t"][j]
        surf    = data_j["surf"][j]
        factu   = data_j["factu"][j]
        densit  = data_j["n"][j]
        radius  = data_j['rloc']
        
        if False:
            x, y, z    = data_j['x'][j][:,0], data_j['x'][j][:,1], data_j['x'][j][:,2]
            
            _slice = np.logical_and((z < radius/20), (z > -radius/20))
            print(np.sum(_slice),_slice.shape[0])
            render(_time, x[_slice], y[_slice], np.log10(densit[_slice]), _id)
            #render(_time, x, y, np.log10(densit), _id)
            
        mask = factu < 1
        f = np.sum(mask)/factu.shape[0]
        if np.sum(mask) == 0:
            continue

        densit = densit[mask]
        factu  = factu [mask]

        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 6))
        
        if len(sys.argv) == 1:
            num_bins = 80
        elif sys.argv[1] == "-e":
            num_bins = np.sum(mask) // 10
            print("bins = ", num_bins)
        if num_bins == 0:
            plt.close(fig)
            continue

        n_ref, mean_vec, median_vec, ten_vec, sample_size = red_to_den(factu, densit)
        ax0.hist(factu, num_bins, density = True)
        ax0.set_xlabel('Reduction factor', fontsize = FONTSIZE)
        ax0.set_ylabel('PDF', fontsize = FONTSIZE)
        ax0.set_xlim(-0.1, 1.1)
        #ax0.set_ylim(-0.1, 4.6)
        ax0.set_title(f'$t$ = {round(_time, 5)}  Myrs\nPoints: ' + f'$n_c > 10^{_id[0]}$ ' + r'$\rm{cm}^{-3}$', fontsize = FONTSIZE)
        ax0.grid(True, which='both', alpha=ALPHA)
        plt.setp(ax0.get_xticklabels(), fontsize = FONTSIZE)
        plt.setp(ax0.get_yticklabels(), fontsize = FONTSIZE)

        if True: # add a little window with the
            x, y    = data_j["x"][j][:, 0], data_j["x"][j][:, 1]
            axins = inset_axes(ax0, width="40%", height="40%", loc='upper left')
            axins.yaxis.set_label_position('right')
            axins.yaxis.tick_right()
            axins.set_xlabel(r"x [Pc]", fontsize=FONTSIZE-2)
            axins.set_ylabel(r"y [Pc]", fontsize=FONTSIZE-2)
            axins.scatter(x, y, marker='o', alpha=ALPHA, s = 0.5)
            axins.set_xlim(-radius*1.2, radius*1.2)
            axins.set_ylim(-radius*1.2, radius*1.2)
            axins.grid(True, which='both', alpha=0.5)
            #axins.set_xticks([])
            #axins.set_yticks([])

        ax1.plot(n_ref, mean_vec, label='mean', linewidth=1.5, linestyle='-', color='darkorange')
        ax1.plot(n_ref, median_vec, label='median', linewidth=1.5, linestyle='--', color='darkorange')
        ax1.plot(n_ref, ten_vec, label='10th percentile', linewidth=1.5, linestyle='-', color='royalblue')
        
        ax1.scatter(densit, factu, marker ='x', color='dimgrey',alpha=0.3)
        
        ax1.set_xscale('log')
        ax1.set_ylabel(r'$R$', fontsize = FONTSIZE)
        ax1.set_xlabel('$n_g$ [cm$^{-3}$]', fontsize = FONTSIZE)
        ax1.set_title(r'$f_{<1}$'+f' = {round(f, 5)}\n Points: ' + f'$r_c < {np.round(radius,4)}$ ' + r'$\rm{Pc}$', fontsize = FONTSIZE)
        ax1.set_ylim(-0.1, 1.1)
        ax1.set_xlim(n_min, n_max)
        plt.setp(ax1.get_xticklabels(), fontsize = FONTSIZE)
        plt.setp(ax1.get_yticklabels(), fontsize = FONTSIZE)
        ax1.grid(True, which='both', alpha=ALPHA)
        ax1.legend()

        plt.savefig(f'./series/{_id}/map_rvn_mosaic{j}.png')
        print(f'saved ./series/{_id}/map_rvn_mosaic{j}.png')
        plt.close(fig)

    plt.close(fig)

    _animate(_id)

if "e-" in sys.argv:
    exit()

fig, axd = plt.subplot_mosaic(
    [["left", "middle", "right"]],
    sharey=True,
    gridspec_kw={"width_ratios": [1, 1, 1], "wspace": 0.05}
)
title = r"Mean ratio $N_{los}/N_{crs}$"

#if len(sys.argv) == 1:
    
ratio0 = np.array([np.mean(nlos0/ncrs) for (nlos0,ncrs) in zip(data2i0["Nlos0"],data2i0["Ncrs"])]) 
ratio1 = np.array([np.median(nlos1/ncrs) for (nlos1,ncrs) in zip(data2i0["Nlos0"],data2i0["Ncrs"])])
ratio2 = np.array([np.std(nlos0/ncrs) for (nlos0,ncrs) in zip(data2i0["Nlos0"],data2i0["Ncrs"])])

texto = r"$r^{ideal}_{cloud} = 0.1 \ \rm{pc}$ "
#axd["left"].plot(data2i0['t'], np.ones_like(data2i0['t']), '--', color="black")
axd["left"].text(data2i0['t'][0], 1.70, texto, fontsize=12, color="black", ha="left", va="bottom")
axd["left"].plot(data2i0['t'], ratio0, 'o-', label=r"Mean $N_{los}/N_{crs}$")
axd["left"].plot(data2i0['t'], ratio1, '.-', label=r"Median $N_{los}/N_{crs}$")
axd["left"].plot(data2i0['t'], ratio0 + ratio2, '--', color=COLORS[1], label="$\sigma$-band")
axd["left"].plot(data2i0['t'], ratio0 - ratio2, '--', color=COLORS[1])
axd["left"].set_ylabel(r"$N_{los}/N_{crs}$",fontsize=FONTSIZE)
axd["left"].set_xlabel("Time [Myrs]",fontsize=FONTSIZE)
axd["left"].grid(True, which='both', alpha=GRID_ALPHA)
axd["left"].legend(frameon = False)

ratio0 = np.array([np.mean(nlos0/ncrs) for (nlos0,ncrs) in zip(data2i1["Nlos0"],data2i1["Ncrs"])]) 
ratio1 = np.array([np.median(nlos1/ncrs) for (nlos1,ncrs) in zip(data2i1["Nlos0"],data2i1["Ncrs"])])
ratio2 = np.array([np.std(nlos0/ncrs) for (nlos0,ncrs) in zip(data2i1["Nlos0"],data2i1["Ncrs"])])

texto = r"$r^{ideal}_{cloud} = 0.2 \ \rm{pc}$"
axd["middle"].text(data2i1['t'][0], 1.70, texto, fontsize=12, color="black", ha="left", va="bottom")
#axd["middle"].plot(data2i1['t'], np.ones_like(data2i1['t']), '--', color="black")
axd["middle"].plot(data2i1['t'], ratio0, 'o-', label="Mean ratio")
axd["middle"].plot(data2i1['t'], ratio1, '.-', label="Median ratio")
axd["middle"].plot(data2i1['t'], ratio0 + ratio2, '--', color=COLORS[1])
axd["middle"].plot(data2i1['t'], ratio0 - ratio2, '--', color=COLORS[1])
axd["middle"].set_xlabel("Time [Myrs]",fontsize=FONTSIZE)
axd["middle"].grid(True, which='both', alpha=GRID_ALPHA)


ratio0 = np.array([np.mean(nlos0/ncrs) for (nlos0,ncrs) in zip(data2i2["Nlos0"],data2i2["Ncrs"])]) 
ratio1 = np.array([np.median(nlos1/ncrs) for (nlos1,ncrs) in zip(data2i2["Nlos0"],data2i2["Ncrs"])])
ratio2 = np.array([np.std(nlos0/ncrs) for (nlos0,ncrs) in zip(data2i2["Nlos0"],data2i2["Ncrs"])])

texto = r"$r^{ideal}_{cloud} = 0.5 \ \rm{pc}$"
axd["right"].text(data2i2['t'][0], 1.70, texto, fontsize=12, color="black", ha="left", va="bottom")
axd["right"].plot(data2i2['t'], ratio0, 'o-', label="Mean ratio")
axd["right"].plot(data2i2['t'], ratio1, '.-', label="Median ratio")
axd["right"].plot(data2i2['t'], ratio0 + ratio2, '--', color=COLORS[1] )
axd["right"].plot(data2i2['t'], ratio0 - ratio2, '--', color=COLORS[1])
axd["right"].set_xlabel("Time [Myrs]",fontsize=FONTSIZE)
axd["right"].tick_params(labelleft=False)
axd["right"].grid(True, which='both', alpha=GRID_ALPHA)
fig.suptitle(title)
plt.savefig( "./series/" + 'ratios012.png', dpi=300)
plt.close(fig)

# first three cases in a single row
fig, axd = plt.subplot_mosaic(
    [["left", "right"]],
    sharey=True,
    gridspec_kw={"width_ratios": [1, 1], "wspace": 0.05}
)
title = r"Ratio $N_{los}/N_{crs}$"

ratio0 = np.array([np.mean(nlos0/ncrs) for (nlos0,ncrs) in zip(data4i3["Nlos0"],data4i3["Ncrs"])]) 
ratio1 = np.array([np.median(nlos1/ncrs) for (nlos1,ncrs) in zip(data4i3["Nlos0"],data4i3["Ncrs"])])
ratio2 = np.array([np.std(nlos0/ncrs) for (nlos0,ncrs) in zip(data4i3["Nlos0"],data4i3["Ncrs"])])

#axd["left"].plot(data2i0['t'], np.ones_like(data2i0['t']), '--', color="black")
axd["left"].plot(data4i3['t'], ratio0, 'o-', label=r"Mean $N_{los}/N_{crs}$")
axd["left"].plot(data4i3['t'], ratio1, '.-', label=r"Median $N_{los}/N_{crs}$")
axd["left"].plot(data4i3['t'], ratio0 + ratio2, '--', color=COLORS[1], label="$\sigma$-band")
axd["left"].plot(data4i3['t'], ratio0 - ratio2, '--', color=COLORS[1])
axd["left"].set_ylabel(r"$N_{los}/N_{crs}$",fontsize=FONTSIZE)
axd["left"].set_ylim(-0.0,2.5)
axd["left"].set_xlabel("Time [Myrs]",fontsize=FONTSIZE)
axd["left"].grid(True, which='both', alpha=GRID_ALPHA)
axd["left"].legend(frameon = False)

ratio0 = np.array([np.mean(nlos0/ncrs) for (nlos0,ncrs) in zip(data6i4["Nlos0"],data6i4["Ncrs"])]) 
ratio1 = np.array([np.median(nlos1/ncrs) for (nlos1,ncrs) in zip(data6i4["Nlos0"],data6i4["Ncrs"])])
ratio2 = np.array([np.std(nlos0/ncrs) for (nlos0,ncrs) in zip(data6i4["Nlos0"],data6i4["Ncrs"])])

#axd["right"].plot(data2i2['t'], np.ones_like(data2i2['t']), '--', color="black")
axd["right"].plot(data6i4['t'], ratio0, 'o-', label="Mean ratio")
axd["right"].plot(data6i4['t'], ratio1, '.-', label="Median ratio")
axd["right"].plot(data6i4['t'], ratio0 + ratio2, '--', color=COLORS[1] )
axd["right"].plot(data6i4['t'], ratio0 - ratio2, '--', color=COLORS[1])
axd["right"].set_ylim(-0.0,2.5)
axd["right"].set_xlabel("Time [Myrs]",fontsize=FONTSIZE)
axd["right"].tick_params(labelleft=False)
axd["right"].grid(True, which='both',  alpha=GRID_ALPHA)
fig.suptitle(title)
plt.savefig( "./series/" + 'ratios34.png', dpi=300)
plt.close(fig)

"""Scatter plots 5 x 5"""

n_plots = 5

fig, ax = plt.subplots()

ax.set_xlabel(r"$n_g$ [cm$^{-3}$]", fontsize=FONTSIZE)
ax.set_ylabel(r"$n_{local}/ n_{ism}$ [adimensional]", fontsize=FONTSIZE)
ax.grid(True, which='both', alpha=GRID_ALPHA)
ax.set_xscale('log')
ax.set_ylim(-0.1, 1.1)
ax.set_xlim(10**2, 10**10)

cmap = plt.get_cmap("viridis")

# Plot lines with colors
for i, (n_val, rt_val) in enumerate(zip(data2i0["n"][-1], data2i0["factu"][-1])):

    _l_ = rt_val < 1
    rt_val = rt_val[_l_]
    n_val = n_val[_l_]
    ax.scatter(n_val, rt_val, marker='x', color="red" ,alpha=0.6)

fig.tight_layout()
plt.savefig("./series/" + 'rfdenscatter.png', dpi=300)
plt.close(fig)

print(data2i0["t"][::5]) # [3.0125     3.625      4.25       4.55097656 4.55196533]
print(data2i1["t"][::5]) # [3.0125     3.625      4.25       4.55097656 4.55196533]
print(data2i2["t"][::5]) # [3.0125     3.625      4.25       4.55097656 4.55196533]
print(data4i3["t"][::5]) # [3.0125     3.625      4.25       4.55097656 4.55196533]
print(data6i4["t"][::5]) # [4.375      4.5515625  4.55201721]

five_sample_mask = np.zeros(len(data2i0["t"]), dtype=bool)
five_sample_mask[::5] = True

t_min = np.min([data2i0["t"].min(), data2i1["t"].min(), data2i2["t"].min()])
t_max = np.max([data2i0["t"].max(), data2i1["t"].max(), data2i2["t"].max()])

fig, axs = plt.subplots(
    3, 3,
    sharey=True,
    sharex=True,
    gridspec_kw={"wspace": 0.0, "hspace": 0.0},
    figsize=(12,12)
)

for j, data in enumerate([data2i0, data2i1, data2i2]):
    time   = data["t"][five_sample_mask]  
    surf   = data["surf"][five_sample_mask]  
    factu  = data["factu"][five_sample_mask]
    densit = data["n"][five_sample_mask]
    radius = data['rloc']
    print(radius)
    
    ax = axs[j,0]

    ax.set_xlabel(r"$n_g$ [cm$^{-3}$]", fontsize=FONTSIZE)
    ax.set_ylabel(r"$R$ [adimensional]", fontsize=FONTSIZE)
    ax.grid(True, which='both', alpha=GRID_ALPHA)
    ax.set_xscale('log')
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlim(10**2, 10**9)
    ax.text(10**(7), 1.0, f"$t = {time[0]}$", fontsize=12, color="red", ha="left", va="bottom")
    ax.text(10**(7), 0.9, f"$f = {surf[0]}$", fontsize=12, color="red", ha="left", va="bottom")
    ax.text(10**(7), 0.8, r"$r_{c}" +f"= {radius}$", fontsize=12, color="red", ha="left", va="bottom")
    n_val  = densit[0]
    rt_val = factu[0] 

    mask = rt_val < 1
    rt_val = rt_val[mask]
    n_val = n_val[mask]

    ax.scatter(n_val, rt_val, marker='x', color="black", alpha=0.3, s =1)

    ax = axs[j,1]

    ax.set_xlabel(r"$n_g$ [cm$^{-3}$]", fontsize=FONTSIZE)
    ax.grid(True, which='both', alpha=GRID_ALPHA)
    ax.set_xscale('log')
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlim(10**2, 10**9)
    ax.text(10**(7), 1.0, f"$t = {time[2]}$", fontsize=12, color="red", ha="left", va="bottom")
    ax.text(10**(7), 0.9, f"$f = {surf[2]}$", fontsize=12, color="red", ha="left", va="bottom")
    ax.text(10**(7), 0.8, r"$r_{c}" +f"= {radius}$", fontsize=12, color="red", ha="left", va="bottom")

    n_val  = densit[2]
    rt_val = factu[2] 

    mask = rt_val < 1
    rt_val = rt_val[mask]
    n_val = n_val[mask]

    ax.scatter(n_val, rt_val, marker='x', color="black", alpha=0.3, s =1)

    ax = axs[j,2]

    ax.set_xlabel(r"$n_g$ [cm$^{-3}$]", fontsize=FONTSIZE)
    ax.grid(True, which='both', alpha=GRID_ALPHA)
    ax.set_xscale('log')
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlim(10**2, 10**9)
    ax.text(10**(7), 1.0, f"$t = {round(time[4],2)}$", fontsize=12, color="red", ha="left", va="bottom")
    ax.text(10**(7), 0.9, f"$f = {surf[4]}$", fontsize=12, color="red", ha="left", va="bottom")
    ax.text(10**(7), 0.8, r"$r_{c}" +f"= {radius}$", fontsize=12, color="red", ha="left", va="bottom")
    
    n_val  = densit[4]
    rt_val = factu[4] 

    mask = rt_val < 1
    rt_val = rt_val[mask]
    n_val = n_val[mask]

    ax.scatter(n_val, rt_val, marker='x', color="black", alpha=0.3, s =1)

plt.savefig(f"./series/rfdenscatter_mosaic.png", dpi=300)
plt.close(fig)

# Plotting individual distributions
# individual plots per dataset
for __, data in enumerate(datas): 
    directory = "./series/" + data["id"] + '/'
    print("\nOutput will go into: ", directory)
    
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

    title = r"Ratio $N_{los}/N_{crs}$"

    axd["left"].plot(data['t'], ratio0, '.-')
    axd["left"].set_ylabel("$N_{los}/N_{crs}$",fontsize=FONTSIZE)
    axd["left"].set_xlabel("Time [Myrs]",fontsize=FONTSIZE)
    axd["left"].grid(True, which='both', alpha=0.5)
    axd["left"].set_ylim(0.5, 2.0)

    axd["right"].plot((data['t']-x1)*1_000, ratio0, ".-")
    axd["right"].set_xlabel("Time [kyrs]",fontsize=FONTSIZE)
    axd["right"].set_xlim((x1-x1)*1_000,(x2-x1)*1_000)
    axd["right"].tick_params(labelleft=False)
    axd["right"].grid(True, which='both', alpha=0.5)
    axd["right"].set_ylim(0.5, 2.0)
    fig.suptitle(title)
    plt.savefig(directory + 'ratio0.png', dpi=300)
    #plt.show()
    plt.close(fig)

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
    axd["left"].set_ylim(0.5, 2.0)

    axd["right"].plot((data['t']-x1)*1_000, ratio1, ".-")
    axd["right"].set_xlabel("Time [kyrs]",fontsize=FONTSIZE)
    axd["right"].set_xlim((x1-x1)*1_000,(x2-x1)*1_000)
    axd["right"].tick_params(labelleft=False)
    axd["right"].grid(True, which='both', alpha=0.5)
    axd["right"].set_ylim(0.5, 2.0)
    fig.suptitle(title)
    plt.savefig(directory + 'ratio1.png', dpi=300)
    #plt.show()
    plt.close(fig)

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

    fig, ax = plt.subplots()

    ax.set_xlabel(r"Time [Myrs]", fontsize=FONTSIZE)
    ax.set_ylabel(r"$n_{local}/n_{ism}$ [adimensional]", fontsize=FONTSIZE)
    ax.grid(True, which='both',alpha=GRID_ALPHA)

    ax.plot(data['t'], f_mean, '-', color='black', alpha=ALPHA, label=r'Mean')
    ax.plot(data['t'], f_median, '--', color='black', alpha=ALPHA, label=r'Median')
    ax.plot(data['t'], f_less, '.-', color='orange', alpha=ALPHA, label=r'$f_{R<1}$')

    ax.legend(fontsize=FONTSIZE)

    percentiles = [10, 20, 30, 40]

    for ptile in percentiles[::-1]:

        f_ptile_down = np.array([np.percentile(ul[ul<1], ptile) for ul in data["factu"]])
        f_ptile_up   = np.array([np.percentile(ul[ul<1], 100-ptile) for ul in data["factu"]])

        ax.fill_between(
            data['t'],
            f_ptile_down,
            f_ptile_up,
            color=COLORS[0],
            alpha=0.3,
            zorder=1
        )

        ax.text(data['t'][4], f_ptile_up[4], f"P{100-ptile}",
                fontsize=6, color="black", ha="left", va="bottom")

        ax.text(data['t'][4], f_ptile_down[4], f"P{ptile}",
                fontsize=6, color="black", ha="left", va="bottom")

    ax.set_ylim(-0.1, 1.1)

    plt.savefig(directory + 'rf_stats.png', dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots()

    ax.set_xlabel(r"Time [Myrs]", fontsize=FONTSIZE)
    ax.grid(True, which='both', alpha=GRID_ALPHA)

    ax.plot(data['t'], f_skew, '-', color='black', alpha=ALPHA, label=r'$\gamma$ : Skewness')
    ax.plot(data['t'], f_kurt, '--', color='black', alpha=ALPHA, label=r'$\kappa$ : Kurtosis')

    ax.legend(fontsize=FONTSIZE-2)

    # inset
    axins = inset_axes(ax, width="40%", height="40%", loc='upper right')

    axins.set_xlabel(r"Time [kyrs]", fontsize=FONTSIZE)

    axins.plot((data['t']-x1)*1000, f_skew, '-', color='black', alpha=ALPHA)
    axins.plot((data['t']-x1)*1000, f_kurt, '--', color='black', alpha=ALPHA)

    axins.scatter((data['t']-x1)*1000, f_skew, color='black', alpha=ALPHA, s=SIZE)
    axins.scatter((data['t']-x1)*1000, f_kurt, color='black', alpha=ALPHA, s=SIZE)

    at_colapse = data['t'] > x1

    y1 = min(np.min(f_skew[at_colapse]), np.min(f_kurt[at_colapse])) * 1.5
    y2 = max(np.max(f_skew[at_colapse]), np.max(f_kurt[at_colapse])) * 1.5

    axins.set_xlim(0, (x2-x1)*1000)
    axins.set_ylim(y1, y2)

    axins.grid(True, which='both', alpha=0.5)

    plt.savefig(directory + 'rf_skew_kurt.png', dpi=300)
    plt.close(fig)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.set_xlabel(r"Time [Myrs]", fontsize=FONTSIZE)
    ax1.set_ylabel(r"$n_{local}/n_{ism}$ [adimensional]", fontsize=FONTSIZE)
    ax1.grid(True, which='both', alpha=GRID_ALPHA)
    ax1.plot(data['t'], f_mean, '-', color='black', alpha=ALPHA, label=r'Mean')
    ax1.plot(data['t'], f_median, '--', color='black', alpha=ALPHA, label=r'Median')
    ax1.plot(data['t'], f_less, '.-', color='orange', alpha=ALPHA, label=r'$f_{R<1}$')
    ax1.legend(frameon = False, fontsize=FONTSIZE-2)

    percentiles = [10, 20, 30, 40]  #percentiles = np.linspace(0,37.5,4) 

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
    axins.grid(True, which='both', alpha=GRID_ALPHA)

    plt.savefig(directory + 'rfstats.png', dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots()

    n_plots = 1

    ax.set_xlabel(r"$n_g$ [cm$^{-3}$]", fontsize=FONTSIZE)
    ax.set_ylabel(r"$n_{local}/ n_{ism}$ [adimensional]", fontsize=FONTSIZE)
    ax.grid(True, which='both', alpha=GRID_ALPHA)
    ax.set_xscale('log')
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlim(10**2, 10**10)

    cmap = plt.get_cmap("viridis")
    t_min, t_max = min(data["t"][::n_plots]), max(data["t"][::n_plots])

    # Plot lines with colors
    for i, (t_val, rt_val) in enumerate(zip(data["t"][::n_plots], data["factu"][::n_plots])):

        _l_ = rt_val < 1
        rt_val = rt_val[_l_]
        n_val = data["n"][i][_l_]
        n_ref, mean_vec, median_vec, ten_vec, sample_size = red_to_den(rt_val, n_val)
        normalized_t = (t_val - t_min) / (t_max - t_min)        
        color = cmap(normalized_t)
        ax.plot(n_ref, mean_vec, '-', lw=2, color=color, alpha=0.6)


    norm = mpl.colors.Normalize(vmin=t_min, vmax=t_max)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Time [Myrs]', fontsize=FONTSIZE)

    fig.tight_layout()
    plt.savefig(directory + 'rfdendistro.png', dpi=300)
    plt.close(fig)

    # plot only after t > x1
    fig, ax = plt.subplots()

    n_plots = 1

    ax.set_xlabel(r"$n_g$ [cm$^{-3}$]", fontsize=FONTSIZE)
    ax.set_ylabel(r"$n_{local}/ n_{ism}$ [adimensional]", fontsize=FONTSIZE)
    ax.grid(True, which='both', alpha=GRID_ALPHA)
    ax.set_xscale('log')
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlim(10**2, 10**10)

    cmap = plt.get_cmap("viridis")
    t_mask = data["t"] > x1
    t_kyrs = (data["t"][t_mask][::n_plots] - x1)*1_000
    t_min, t_max = 0.0, (x2-x1)*1_000

    # Plot lines with colors
    for i, (t_val, rt_val) in enumerate(zip(t_kyrs, data["factu"][t_mask][::n_plots])):

        _l_ = rt_val < 1
        rt_val = rt_val[_l_]
        n_val = data["n"][t_mask][i][_l_]

        n_ref, mean_vec, median_vec, ten_vec, sample_size = red_to_den(rt_val, n_val)
        normalized_t = (t_val) / (t_max)        
        color = cmap(normalized_t)
        ax.plot(n_ref, mean_vec, '-', lw=2, color=color, alpha=0.6)
        #ax.set_xlim(10**2, 10**10)

    norm = mpl.colors.Normalize(vmin=t_min, vmax=t_max)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label(r'$t - t_0$ [kyrs]', fontsize=FONTSIZE)

    fig.tight_layout()
    plt.savefig(directory + 'rfdendistrokyrs.png', dpi=300)
    plt.close(fig)



