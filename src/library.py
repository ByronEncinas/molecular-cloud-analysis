import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import spatial
from scipy.spatial import cKDTree
from scipy import interpolate
import matplotlib as mpl
from functools import wraps

mpl.rcParams['text.usetex'] = False

""" Toggle Parameters """

""" Statistics Methods """

def red_to_den(factor: np.array, numb: np.array, criteria = 1/8): # vectorize

    total = factor.shape[0]

    n_ref = np.logspace(np.log10(numb.min()), np.log10(numb.max()), total)
    log_numb = np.log10(numb)          # shape (M,)  — compute ONCE outside loop
    log_n_ref   = np.log10(n_ref)           # shape (N,)

    # mask with boolean array, matching the condition log10(n_g(p)/n_{ref}) < 1/8
    #print(log_numb.shape, log_n_ref.shape)
     # <- this creates a matrix such that each row, can be collapse to a mean, median and percentile
    mask = np.abs(log_numb[np.newaxis, :] - log_n_ref[:, np.newaxis]) < (1/8)
    #print(mask.shape)
    # broadcast mask to factor[np.newaxis, :] or np.nan
    meet_condition = np.where(mask, factor[np.newaxis, :], np.nan)

    import warnings

    with warnings.catch_warnings(): 
        # following error messages are not relevant, they just point to a empty slice
        warnings.filterwarnings("ignore", message="Mean of empty slice") 
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")

        # mean function that ignores nan values
        mean_vec = np.nanmean(meet_condition, axis=1)
        median_vec = np.nanmedian(meet_condition, axis=1)
        ten_vec = np.nanquantile(meet_condition, 0.1, axis=1)
        sample_size = mask.sum(axis=1).astype(int)
    
    return n_ref, mean_vec, median_vec, ten_vec, sample_size

def reduction_to_density(factor, numb):
    factor = np.array(factor)
    numb = np.array(numb)
    # R is numpy array
    def match_ref(n, d_data, r_data, p_data=0):
        sample_r = []

        for i in range(0, len(d_data)):
            if np.abs(np.log10(d_data[i]/n)) < 1/8:
                sample_r.append(r_data[i])
        sample_r.sort()
        try:
            mean = sum(sample_r)/len(sample_r)
            median = np.quantile(sample_r, .5)
            ten = np.quantile(sample_r, .1)
            size = len(sample_r)
        except:
            return [sample_r*0, np.nan, np.nan, np.nan, 0]
        return [sample_r, mean, median, ten, size]

    total = factor.shape[0]
    x_n = np.logspace(np.min(np.log10(numb)), np.max(np.log10(numb)), total)
    mean_vec = np.zeros(total)
    median_vec = np.zeros(total)
    ten_vec = np.zeros(total)
    sample_size = np.zeros(total)
    matrix  = []
    for i in range(0, total):
        s = match_ref(x_n[i], numb, factor)
        matrix+=[[s[0]]]
        mean_vec[i] = s[1]
        median_vec[i] = s[2]
        ten_vec[i] = s[3]
        sample_size[i] = s[4]
    
    return x_n, matrix, mean_vec, median_vec, ten_vec, sample_size

def get_globals_memory() -> None:
    import sys

    total = 0
    for name, obj in globals().items():
        if name.startswith("__") and name.endswith("__"):
            continue  # skip built-in entries
        try:
            total += sys.getsizeof(obj)
        except TypeError:
            pass  # some objects might not report size

    # Convert bytes → gigabytes
    gb = total / (1024 ** 3)
    print(f"Memory used by globals: {gb:.6f} gigabytes")

""" Constants and convertion factor """

# Unit Conversions
gauss_to_micro_gauss = 1e+6
km_to_parsec = 1 / 3.085677581e13  # 1 pc in km
pc_to_cm = 3.086 * 1.0e+18  # cm per parsec
AU_to_cm = 1.496 * 1.0e+13  # cm per AU
surface_to_column = 2.55e+23
pc_myrs_to_km_s = 0.9785

# Physical Constants
mass_unit = 1.99e33  # g
length_unit = 3.086e18  # cm in a parsec
velocity_unit = 1e5  # cm/s
time_unit = length_unit / velocity_unit  # s
seconds_in_myr = 1e6 * 365.25 * 24 * 3600  # seconds in a million years (Myr)
boltzmann_constant_cgs = 1.380649e-16  # erg/K
grav_constant_cgs = 6.67430e-8  # cm^3/g/s^2
hydrogen_mass = 1.6735e-24  # g

# Code Units Conversion Factors
myrs_to_code_units = seconds_in_myr / time_unit
code_units_to_gr_cm3 = 6.771194847794873e-23  # conversion from code units to g/cm^3
gauss_code_to_gauss_cgs = (4 * np.pi)**0.5   * (3.086e18)**(-1.5) * (1.99e33)**0.5 * 1e5 # cgs units

# ISM Specific Constants
mean_molecular_weight_ism = 2.35  # mean molecular weight of the ISM (Wilms, 2000)
gr_cm3_to_nuclei_cm3 = 6.02214076e+23 / (2.35) * 6.771194847794873e-23  # Wilms, 2000 ; Padovani, 2018 ism mean molecular weight is # conversion from g/cm^3 to nuclei/cm^3

# cache-ing spatial.cKDTree(Pos[:]).query(x, k=1)
_cached_tree = None
_cached_pos = None

""" Extra """

def field_lines_r_vol(b, r, r0, lr):
    from gists.__vor__ import traslation_rotation
    from copy import deepcopy
    # b   fields
    # r   vectors
    # r0  x_input

    # select line
    m = int(sys.argv[-1]) 
    zoom = 20
    field = b[:,m]
    vector = r[:,m,:]

    # track
    x0, y0, z0 = r0[m]
    x, y, z = vector[:,0] - x0, vector[:,1]-y0, vector[:,2]-z0

    mk = np.logical_and(field > 0, x**2 + y**2 + z**2 < (zoom)**2)
    x, y, z = x[mk], y[mk], z[mk]
    field = field[mk] 
    vector = vector[mk,:]

    bhat = field/np.max(field)

    fig, axd = plt.subplot_mosaic([["profile", "field3d"]], figsize=(10, 4))
    arg_input = np.where(vector[:,0] == r0[m,0])[0][0]
    ax3d = fig.add_subplot(122, projection="3d")
    
    norm = LogNorm(vmin=np.min(bhat), vmax=np.max(bhat))
    cmap = cm.viridis

    for l in range(len(x) - 1):
        color = cmap(norm(bhat[l]))
        ax3d.plot(x[l:l+2], y[l:l+2], z[l:l+2], color=color, linewidth=2)
    
    ax3d.scatter(0,0,0, marker="x", color="g", s=6)
    dc = np.array([np.diff(x[arg_input:arg_input+2]), np.diff(y[arg_input:arg_input+2]),np.diff(z[arg_input:arg_input+2])])
    dc /= np.linalg.norm(dc, axis=0) * 1.e+2

    n = 15
    a_ =  6.e-3
    b_ = -a_

    X, Y, Z = np.meshgrid(np.linspace(a_,b_, n), np.linspace(a_,b_, n), np.linspace(a_,b_, n))
    mk = (X**2 + Y**2 + Z**2) < a_**2
    X, Y, Z = X[mk], Y[mk], Z[mk]
    points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    _x = np.array([0.0, 0.0, 0.0])
    _b = deepcopy(dc)
    p = traslation_rotation(_x, _b[:,0], points)
    ax3d.scatter(p[:,0], p[:,1], p[:,2], marker='x', c='g', alpha=0.3, s=2)

    ax3d.view_init(elev=90, azim=00)
    #ax3d.set_xlim(-0.01,0.01)
    #ax3d.set_ylim(-0.01,0.01)
    #ax3d.set_zlim(-0.01,0.01)
    ax3d.set_xlabel("x [pc]")
    ax3d.set_ylabel("y [pc]")
    ax3d.set_zlabel("z [pc]")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax3d, shrink=0.8)
    cbar.set_label("$\mu$-G")
    field = b[:,m]
    __ = field>0
    field = field[__]
    vector = r[__,m,:]
    displacement = np.cumsum(np.linalg.norm(np.diff(vector, axis=0), axis=1))
    arg_input = np.where(vector[:,0] == r0[m,0])[0][0]
    ax = axd["profile"]
    ax.set_xlabel("Displacement")
    ax.set_ylabel("B [$\mu$G]")
    ax.plot(displacement, field[1:], "-", label="B")

    ax.scatter(displacement[arg_input+1], field[arg_input],c='black',  label="x")
    ax.set_xlim(displacement[arg_input-600], displacement[arg_input+600])
    ax.set_yscale('log')
    ax.legend()
    

    plt.tight_layout()
    plt.savefig("./field_mosaic.png", dpi=300)
    plt.close(fig)
    
def field_lines_norm(b, r, r0):
    
    m = r.shape[1]
    
    elevation = 0
    azimuth   = 0
    zoom      = 0.1     # axis zoom
    zoom2     = 2.0*zoom # spherical window zoom
    """
    r  /= AU_to_cm
    r0 /= AU_to_cm
    r_rxb_z = []
    for k in range(m):
        x0=r0[k, 0]
        y0=r0[k, 1]
        z0=r0[k, 2]
        x=r[:,k, 0]
        y=r[:,k, 1]
        z=r[:,k, 2]
        mk0  = b[:, k] > 0
        mk1  = (x*x + y*y + z*z) < (zoom2)**2
        mk = np.logical_and(mk0,mk1)
        diff_ = np.diff(vectors[mk,k,:], axis=0)
        rxb = np.cross(vectors[mk,k,:][1:,:], diff_, axis=1)
        r_rxb = np.cross(vectors[mk,k,:][1:,:], rxb, axis=1)
        r_rxb_z += [np.mean(r_rxb,axis=0)]
        
    r_rxb_z = np.mean(r_rxb_z, axis=0)*0.1/np.linalg.norm(np.mean(r_rxb_z, axis=0))

    from gists.__vor__ import traslation_rotation

    n = 50
    a_ =  zoom2*0.8
    b_ = -a_

    X, Y = np.meshgrid(np.linspace(a_,b_, n), np.linspace(a_,b_, n)); Z = np.zeros_like(X)
    mk = (X**2 + Y**2) < a_**2 ; X, Y, Z = X[mk], Y[mk], Z[mk]
    points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

    rot_plane = traslation_rotation(np.array([0.0, 0.0, 0.0]), r_rxb_z, points, p=False)
    """
    from matplotlib import cm
    from matplotlib.colors import Normalize, LogNorm

    norm = Normalize(vmin=np.min(b), vmax=np.max(b))
    cmap = cm.viridis

    ax = plt.figure().add_subplot(projection='3d')
    
    for k in range(m):
        x0=r0[k, 0]
        y0=r0[k, 1]
        z0=r0[k, 2]
        ax.scatter(x0, y0, z0, marker="x",color="g",s=6)            
            
    #ax.scatter(rot_plane[:,0], rot_plane[:,1], rot_plane[:,2], marker="x",color="r",s=3, alpha=0.25)            
    #ax.quiver(0,0,0, r_rxb_z[0],r_rxb_z[1],r_rxb_z[2],color="black")

    ax.set_xlim(-zoom*2,zoom*2)
    ax.set_ylim(-zoom*2,zoom*2)
    ax.set_zlim(-zoom*2,zoom*2)
    ax.set_xlabel('x [au]')
    ax.set_ylabel('y [au]')
    ax.set_zlabel('z [au]')
    ax.set_title('Starting Points')
    ax.view_init(elev=elevation, azim=azimuth)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label('Arbitrary Units')
    plt.savefig("./StartingPoints.png", bbox_inches='tight', dpi=300)

    try:

        from matplotlib import cm
        from matplotlib.colors import Normalize

        ax = plt.figure().add_subplot(projection='3d')
        #r /= pc_to_cm

        for k in range(0,m,1):
            # mask makes sure that start and ending point arent the zero vector
            x0=r0[k, 0]
            y0=r0[k, 1]
            z0=r0[k, 2]
            ax.scatter(x0, y0, z0, marker="x",color="black",s=1,alpha=0.05)   
            x=r[:,k, 0]
            y=r[:,k, 1]
            z=r[:,k, 2]
            mk0  = b[:, k] > 0
            mk1  = x*x + y*y + z*z < (zoom2)**2
            mk = np.logical_and(mk0,mk1)
            x=r[mk,k, 0]
            y=r[mk,k, 1]
            z=r[mk,k, 2]
            bhat = b[mk, k]
            bhat /= np.max(bhat)
            norm = LogNorm(vmin=np.min(bhat), vmax=np.max(bhat))
            cmap = cm.viridis

            ax.scatter(x0, y0, z0, marker="x",color="g",s=1, alpha=0.5, label="X")            
            for l in range(len(x) - 1):
                color = cmap(norm(bhat[l]))
                ax.plot(x[l:l+2], y[l:l+2], z[l:l+2], color=color, linewidth=0.3)

        #ax.scatter(rot_plane[:,0], rot_plane[:,1], rot_plane[:,2], marker="x",color="r",s=3, alpha=0.25)            
        #ax.quiver(0,0,0, r_rxb_z[0],r_rxb_z[1],r_rxb_z[2],color="black")
        ax.set_xlim(-zoom*2,zoom*2)
        ax.set_ylim(-zoom*2,zoom*2)
        ax.set_zlim(-zoom*2,zoom*2)
        ax.set_xlabel('x [Pc]')
        ax.set_ylabel('y [Pc]')
        ax.set_zlabel('z [Pc]')
        ax.set_title('Magnetic field morphology')
        ax.view_init(elev=elevation, azim=azimuth)
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
        cbar.set_label('Arbitrary Units')
        

        plt.savefig("./FieldTopology.png", bbox_inches='tight', dpi=300)

    except Exception as e:
        print(e)
        print("Couldnt print B field structure")

def cr_density_on_diff_cloud_densities():

    # Load the pickled DataFrames
    df2 = pd.read_pickle(f'./series/data_dc2.0_{INPUT}.pkl')
    df2.index.name = 'snapshot'
    df2.index = df2.index.astype(int)
    df2 = df2.sort_index()

    df4 = pd.read_pickle(f'./series/data_dc4.0_{INPUT}.pkl')
    df4.index.name = 'snapshot'
    df4.index = df4.index.astype(int)
    df4 = df4.sort_index()
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)

    # import n-arrays 
    t2     = df2["time"].to_numpy()
    factu2 = df2["r_u"].to_numpy()
    n2     = df2["n_rs"].to_numpy()
    #x2     = df2["x_input"].to_numpy()
    #B2     = df2["B_rs"].to_numpy()
    #N2los0 = df2["n_los0"].to_numpy()  # mean
    #N2los1 = df2["n_los1"].to_numpy()  # median
    #N2crs  = df2["n_path"].to_numpy()
    #surf2  = df2["surv_fraction"].to_numpy()
    #factl2 = df2["r_l"].to_numpy()

    t4     = df4["time"].to_numpy()
    factu4 = df4["r_u"].to_numpy()
    n4     = df4["n_rs"].to_numpy()
    #x4     = df4["x_input"].to_numpy()
    #B4     = df4["B_rs"].to_numpy()
    #N4los0 = df4["n_los0"].to_numpy()  # mean
    #N4los1 = df4["n_los1"].to_numpy()  # median
    #N4crs  = df4["n_path"].to_numpy()
    #surf4  = df4["surv_fraction"].to_numpy()
    #factl4 = df4["r_l"].to_numpy()

    n_plots = 1

    fig, ax = plt.subplots(
        1, 2, figsize=(14, 6),
        gridspec_kw={'wspace': 0},
        sharey=True
    )

    #fig, ax = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    output = 'r_n_2_4_'

    cmap = plt.get_cmap("viridis")
    t_min, t_max = 2.9, max(np.max(t4), np.max(t2))
    print(t2.shape, t4.shape)
    print(min(t2), max(t2))
    print(min(t4), max(t4))


    # ---- LEFT PANEL ----
    ax[0].set_xlabel(r"$n_g$ [cm$^{-3}$]", fontsize=16)
    ax[0].set_title(r"$n_{cloud} > 10^4$ cm$^{-3}$", fontsize=16)
    ax[0].set_ylabel(r"$n_{local}/ n_{ism}$", fontsize=16)
    ax[0].grid(True, which='both', alpha=0.3)
    ax[0].set_xscale('log')
    ax[0].set_ylim(-0.1, 1.1)
    ax[0].tick_params(axis='both', labelsize=14)

    for i, (t_val, n_val ,rt_val) in enumerate(zip(t4[::n_plots], n4[::n_plots], factu4[::n_plots])):
        _ = rt_val < 1.
        n_ref, r_matrix, mean_vec, median_vec, ten_vec, sample_size = reduction_to_density(rt_val[_], n_val[_])
        color = cmap((t_val - t_min) / (t_max - t_min))
        ax[0].plot(n_ref, mean_vec, '-', lw=2.0, color=color, alpha=0.6)


    ax[1].set_xlabel(r"$n_g$ [cm$^{-3}$]", fontsize=16)
    ax[1].set_title(r"$n_{cloud} > 10^2$ cm$^{-3}$", fontsize=16)
    ax[1].grid(True, which='both', alpha=0.3)
    ax[1].set_xscale('log')
    ax[1].set_ylim(-0.1, 1.1)
    ax[1].tick_params(axis='both', labelsize=14)

    for i, (t_val, n_val ,rt_val) in enumerate(zip(t2[::n_plots], n2[::n_plots], factu2[::n_plots])):
        _ = rt_val < 1.
        n_ref, r_matrix, mean_vec, median_vec, ten_vec, sample_size = reduction_to_density(rt_val[_], n_val[_])
        color = cmap((t_val - t_min) / (t_max - t_min))
        ax[1].plot(n_ref, mean_vec, '-', lw=2.0, color=color, alpha=0.6)

    norm = mpl.colors.Normalize(vmin=t_min, vmax=t_max)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=ax.ravel().tolist())
    cbar.set_label('Time [Myrs]', fontsize=16)
    cbar.ax.tick_params(labelsize =14)

    plt.savefig('./' + output + f'{INPUT}.png', dpi=300)
    plt.close(fig)
    return None

def local_cr_spectra(size):
    energy = np.logspace(3, 9, size)
    C, alpha, beta, Enot = select_species('L')
    ism_spectrum = lambda x: C*(x**alpha/((Enot + x)**beta))/(4*np.pi)
    log_spec_ism_low  = np.log10(ism_spectrum(energy))
    C, alpha, beta, Enot = select_species('H')
    ism_spectrum = lambda x: C*(x**alpha/((Enot + x)**beta))/(4*np.pi)
    log_spec_ism_high = np.log10(ism_spectrum(energy))
    log_energy = np.log10(energy)

    
    """
    IONIZATION RATE

    \zeta_i(s) = \int_{-1}^{1}d\mu \int_0^{\infty} j(E', \mu, s) \sigma_{ion}(E')dE'
    
    """
    Neff  = np.logspace(19, 27, size) 

    # (Padovani et al 2018) LLR - Long lived radionuclei  
    log_zeta_llr  = np.log10(1.4e-22)*np.ones_like(Neff) 
    log_zeta_std  = np.log10(1.0e-17)*np.ones_like(Neff) 
    log_zeta_low, log_zeta_high = ionization_rate_fit(Neff)
    loss_function = lambda z: Lstar*(Estar/z)**d 
    n_mirr_l_at_x, zeta_l_at_x, loc_spec_l_at_x = x_ionization_rate(fields[0], densities[0], vectors[0], x_input[0], m='L')
    n_mirr_h_at_x, zeta_h_at_x, loc_spec_h_at_x = x_ionization_rate(fields[0], densities[0], vectors[0], x_input[0], m='H')
    

    output = 'ion_lh_'
    
    log_ionization_path_l, log_ionization_path_h = ionization_rate_fit(Ncrs[-1])
    log_ionization_los_l, log_ionization_los_h   = ionization_rate_fit(Nlos0[-1])

    fig, axs = plt.subplots(1, 2, figsize=(5, 5),gridspec_kw={'wspace': 0, 'hspace': 0}, sharey=True)

    _x = min(np.min(Nlos0[-1]), np.min(Ncrs[-1]))*15
    _y = -16.4

    axs[0].text(_x, _y, "$\mathcal{L}$",
        fontsize=20,
        color="black",
        rotation=0,
        ha="center",   # horizontal alignment: left, center, right
        va="bottom") # vertical alignment: top, center, bottom
    #axs[0].set_title("Model $\mathcal{L}$", fontsize=16)
    axs[0].set_xlabel(r'''N [cm$^{-2}$]''', fontsize=16)
    axs[0].set_ylabel(r'$log_{10}(\zeta /\rm{s}^{-1})$', fontsize=16)
    axs[0].set_xscale('log')
    axs[0].scatter(Nlos0[-1], log_ionization_los_l, marker='x', color="#8E2BAF", s=15, alpha=0.6, label=r'$\zeta_{\rm los}$')
    axs[0].scatter(Ncrs[-1], log_ionization_path_l-1, marker='|', color="#148A02", s=15, alpha=0.6, label=r'$\zeta_{\rm path}/10$')
    axs[0].axhline(y=log_zeta_std[0], linestyle='--', color='black', alpha=0.6)
    axs[0].axhline(y=log_zeta_llr[0], linestyle='--', color='black', alpha=0.6)

    axs[0].grid(True, which='both', alpha=0.3)
    axs[0].set_ylim(-22, -16)

    _x = min(np.min(Nlos0[-1]), np.min(Ncrs[-1]))*15

    axs[1].text(_x, _y, "$\mathcal{H}$",
        fontsize=20,
        color="black",
        rotation=0,
        ha="center",   # horizontal alignment: left, center, right
        va="bottom") # vertical alignment: top, center, bottom
    
    axs[1].set_xlabel(r'''$N$ [cm$^{-2}$]''', fontsize=16)
    axs[1].set_xscale('log')
    axs[1].scatter(Nlos0[-1], log_ionization_los_h, marker='x', color="#8E2BAF", s=15, alpha=0.6, label=r'$\zeta_{\rm los}$')
    axs[1].scatter(Ncrs[-1], log_ionization_path_h-1, marker='|', color="#148A02", s=15, alpha=0.6, label=r'$\zeta_{\rm path}/10$')
    axs[1].axhline(y=log_zeta_std[0], linestyle='--', color='black', alpha=0.6)
    axs[1].axhline(y=log_zeta_llr[0], linestyle='--', color='black', alpha=0.6)
    axs[1].grid(True, which='both', alpha=0.3)
    axs[1].set_ylim(-22, -16)
    axs[1].tick_params(labelleft=False)
    axs[1].legend(fontsize=14, loc='lower left')
    #fig.tight_layout()
    plt.savefig('series/' + output + f'{INPUT}.png', dpi=300)
    plt.close(fig)

    output = 'ion_ratio_lh_'

    log_ionization_path_l, log_ionization_path_h = ionization_rate_fit(Ncrs[-1])
    log_ionization_los_l, log_ionization_los_h   = ionization_rate_fit(Nlos0[-1])

    ratio_los_path_l = log_ionization_los_l - log_ionization_path_l
    ratio_los_path_h = log_ionization_los_h - log_ionization_path_h

    fig, axs = plt.subplots()

    _x = min(np.min(Nlos0[-1]), np.min(Ncrs[-1]))*15
    _y = -16.4

    axs.text(_x, _y, "$\mathcal{L}$",
        fontsize=20,
        color="black",
        rotation=0,
        ha="center",   # horizontal alignment: left, center, right
        va="bottom") # vertical alignment: top, center, bottom
    
    axs.set_xlabel(r'''$N$ [cm$^{-2}$]''', fontsize=16)
    axs.set_ylabel(r'$\zeta(N_{los})/\zeta(N_{crs})$', fontsize=16)
    axs.set_xscale('log')
    axs.scatter(Nlos0[-1], ratio_los_path_l, marker='x', color="#8E2BAF", s=15, alpha=0.6, label=r'Model $\mathcal{L}$')
    axs.scatter(Ncrs[-1], ratio_los_path_h, marker='x', color="#148A02", s=15, alpha=0.6, label=r'Model $\mathcal{H}$')
    axs.grid(True, which='both', alpha=0.3)
    #fig.tight_layout()
    plt.savefig('series/' + output + f'{INPUT}.png', dpi=300)
    plt.close(fig)

    # campo en x, mangitud campo en x, densidad en x y ID de la celda
    local_fields_1, abs_local_fields_1, local_densities, cells = find_points_and_get_fields(
        x, Bfield, Density, Density_grad, Pos, VoronoiPos
    )

    # vector unitario en la dirección del campo en x
    local_fields_1 = local_fields_1 / np.tile(abs_local_fields_1, (3, 1)).T

    # Volume de la celda en la que está x
    CellVol = Volume[cells]

    # radio de esfera con volumen CellVol
    scaled_dx = dx * ((3/4) * CellVol / np.pi)**(1/3)

    # paso final
    x_final = x + 0.5 * scaled_dx[:, np.newaxis] * (local_fields_1)

    return x_final, abs_local_fields_1, local_densities, CellVol

""" Ionization rate modules & constants """

size = 10_000
Ei = 1.0e+3
Ef = 1.0e+9
n0 = 150 #cm−3 and 
k  = 0.5 # –0.7
d = 0.82
a = 0.1 # spectral index either 0.1 from Low Model, or \in [0.5, 2.0] according to free streaming analytical solution.
# mean energy lost per ionization event
epsilon = 35.14620437477293
# Luminosity per unit volume for cosmic rays (eV cm^2)
Lstar = 1.4e-14
# Flux constant (eV^-1 cm^-2 s^-1 sr^-1) C*(10e+6)**(0.1)/(Enot+6**2.8)
Jstar = 2.43e+15*(10e+6)**(0.1)/(500e+6**2.8) # Proton in Low Regime (M. Padovani & A. Ivlev 2015) https://iopscience.iop.org/article/10.1088/0004-637X/812/2/135
# Flux constant (eV^-1 cm^-2 s^-1)
Enot = 500e+6
Jstar = 2.4e+15*(1.0e+6)**(0.1)/(Enot**2.8)
# Energy scale for cosmic rays (1 MeV = 1e+6 eV)
Estar = 1.0e+6
logE0, logEf = 6, 9
energy = np.logspace(logE0, logEf, size)
diff_energy = np.array([energy[k]-energy[k-1] for k in range(len(energy))])
diff_energy[0] = energy[0]

model = None

def select_species(m):

    if m == 'L':
        C = 2.43e+15 *4*np.pi
        alpha, beta = 0.1, 2.8
        Enot = 650e+6
    elif m == 'H':
        C = 2.43e+15 *4*np.pi
        alpha, beta = -0.8, 1.9
        Enot = 650e+6
    elif m == 'e':
        C = 2.1e+18*4*np.pi
        alpha, beta = -1.5, 1.7
        Enot = 710e+6
    else:
        raise ValueError(f"[Error] Argument {m} not supported")
    return C, alpha, beta, Enot

# only for protonsexp_data/cross_pH2_rel_1e18.npz
cross_data = np.load('exp_data/cross_pH2_rel_1e18.npz')
loss_data  = np.load('exp_data/pLoss.npz')

cross = interpolate.interp1d( cross_data["E"], cross_data["sigmap"])
loss = interpolate.interp1d(loss_data["E"], loss_data["L_full"])

def ionization_rate_fit(Neff):
    """  
    \mathcal{L} & \mathcal{H} Model: Protons

    """
    model_H = [1.001098610761e7, -4.231294690194e6,  7.921914432011e5,
            -8.623677095423e4,  6.015889127529e3, -2.789238383353e2,
            8.595814402406e0, -1.698029737474e-1, 1.951179287567e-3,
            -9.937499546711e-6
    ]


    model_L = [-3.331056497233e+6,  1.207744586503e+6,-1.913914106234e+5,
                1.731822350618e+4, -9.790557206178e+2, 3.543830893824e+1, 
            -8.034869454520e-1,  1.048808593086e-2,-6.188760100997e-5, 
                3.122820990797e-8]

    logzl = []
    for i,Ni in enumerate(Neff):
        lzl = sum( [cj*(np.log10(Ni))**j for j, cj in enumerate(model_L)] )
        logzl.append(lzl)

    logzh = []

    for i,Ni in enumerate(Neff):
        lzh = sum( [cj*(np.log10(Ni))**j for j, cj in enumerate(model_H)] )
        logzh.append(lzh)

    from scipy import interpolate

    log_L = interpolate.interp1d(Neff, logzl)
    log_H = interpolate.interp1d(Neff, logzh)
    return log_L(Neff), log_H(Neff)

def column_density(radius_vector, magnetic_field, numb_density, direction='', mu_ism = np.logspace(-2, 0, num=50)):
    trajectory = np.cumsum(np.linalg.norm(radius_vector, axis=1)) #np.insert(, 0, 0.0)
    dmui = np.insert(np.diff(mu_ism), 0, mu_ism[0])    
    ds = np.insert(np.linalg.norm(np.diff(radius_vector, axis=0), axis=1), 0, 0.0)
    Nmu  = np.zeros((len(magnetic_field), len(mu_ism)))
    dmu = np.zeros((len(magnetic_field), len(mu_ism)))
    mu_local = np.zeros((len(magnetic_field), len(mu_ism)))
    B_ism     = magnetic_field[0]

    for i, mui_ism in enumerate(mu_ism):

        for j in range(len(magnetic_field)):

            n_g = numb_density[j]
            Bsprime = magnetic_field[j]
            mu_local2 = 1 - (Bsprime/B_ism) * (1 - mui_ism**2)
            
            if (mu_local2 <= 0):
                break
            mu_local[j,i] = np.sqrt(mu_local2)
            dmu[j,i] = dmui[i]*(mui_ism/mu_local[j,i])*(Bsprime/B_ism)
            Nmu[j, i] = Nmu[j - 1, i] + n_g * ds[j] / mu_local[j, i] if j > 0 else n_g * ds[j] / mu_local[j, i]

    return Nmu, mu_local, dmu, trajectory

def mirrored_column_density(radius_vector, magnetic_field, numb_density, Nmu, direction='', mu_ism = np.logspace(-2, 0, num=50)):

    ds    = np.insert(np.linalg.norm(np.diff(radius_vector, axis=0), axis=1), 0, 0.0)
    Nmir  = np.zeros((len(magnetic_field), len(mu_ism)))
    s_max = np.argmax(magnetic_field) 
    if  'fwd' in direction:
        s_max = np.argmax(magnetic_field) + 1

    B_ism = magnetic_field[0]

    for i, mui_ism in enumerate(mu_ism): # cosina alpha_i
        for s in range(s_max):            # at s
            N = Nmu[s, i]
            for s_prime in range(s_max-s): # get mirrored at s_mirr; all subsequent points s < s_mirr up until s_max
                # sprime is the integration variable.
                if (magnetic_field[s_prime] > B_ism*(1-mui_ism**2)):
                    break
                mu_local = np.sqrt(1 - magnetic_field[s_prime]*(1-mui_ism**2)/B_ism )
                s_mir = s + s_prime
                dens  = numb_density[s:s_mir]
                diffs = ds[s:s_mir] 
                N += np.sum(dens*diffs/mu_local)
            Nmir[s,i] = N

    return Nmir

def ionization_rate(Nmu, mu_local, dmu, direction = '',mu_ism = np.logspace(-2, 0, num=50), m='L'):

    C, alpha, beta, Enot = select_species(m)
    ism_spectrum = lambda x: C*(x**alpha/((Enot + x)**beta))
    loss_function = lambda z: Lstar*(Estar/z)**d
    zeta_mui = np.zeros_like(Nmu)
    zeta = np.zeros_like(Nmu[:,0])
    jspectra = np.zeros((Nmu.shape[0], energy.shape[0]))

    for l, mui in enumerate(mu_ism):

        for j, Nj in enumerate(Nmu[:,l]):
            mu_ = mu_local[j,l]
            if mu_ <= 0:
                break

            #  Ei = ((E)**(1 + d) + (1 + d) * Lstar* Estar**(d)* Nj)**(1 / (1 + d)) 
            Ei = ((energy)**(1 + d) + (1 + d) * Lstar* Estar**(d)* Nj/mu_)**(1 / (1 + d))

            isms = ism_spectrum(Ei)                        # log_10(j_i(E_i))
            llei = loss_function(Ei)                       # log_10(L(E_i))
            sigma_ion = cross(energy)
            spectra   = 0.5*isms*llei/loss_function(energy)  
            
            jspectra[j,:] = spectra
            #jl_dE = np.sum(isms*llei*diff_energy)  # j_i(E_i)L(E_i) = 10^{log_10(j_i(E_i)) + log_10(L(E_i))}
            #zeta_mui[j, l] = jl_dE / epsilon 
            zeta_mui[j, l] = np.sum(spectra*sigma_ion*diff_energy)

    zeta = np.sum(dmu * zeta_mui, axis = 1)

    return zeta, zeta_mui, jspectra

def local_spectra(Nmu, mu_local, mu_ism = np.logspace(-2, 0, num=50), m='L'):
    C, alpha, beta, Enot = select_species(m)
    ism_spectrum = lambda x: C*(x**alpha/((Enot + x)**beta))
    loss_function = lambda z: Lstar*(Estar/z)**d 
    jspectra = np.zeros((Nmu.shape[0], energy.shape[0]))

    for l, mui in enumerate(mu_ism):
        for j, Nj in enumerate(Nmu[:,l]):
            mu_ = mu_local[j,l]
            if mu_ <= 0:
                break

            Ei = ((energy)**(1 + d) + (1 + d) * Lstar* Estar**(d)* Nj/mu_)**(1 / (1 + d))

            isms = ism_spectrum(Ei)                        # log_10(j_i(E_i))
            llei = loss_function(Ei)                       # log_10(L(E_i))
            jspectra[j,:] = 0.5*isms*llei/loss_function(energy)  

    return np.sum(jspectra, axis = 0)

def x_ionization_rate(fields, densities, vectors, x_input, m='L'):

    C, alpha, beta, Enot = select_species(m)
    ism_spectrum = lambda x: C*(x**alpha/((Enot + x)**beta))
    loss_function = lambda z: Lstar*(Estar/z)**d

    lines = x_input.shape[0]
    zeta_at_x = np.zeros(lines)
    nmir_at_x = np.zeros(lines)

    local_spectra_at_x = np.zeros((lines, size))

    for line in range(lines):
        density    = densities[:, line]
        field =  fields[:, line]*1e6 # microgauss
        vector  =  vectors[:, line, :]

        # slice out zeroes        
        mask = np.where(density > 0.0)[0]
        start, end = mask[0], mask[-1]

        density    = density[start:end]
        field =  field[start:end] #np.ones_like(field[start:end]) 
        vector  =  vector[start:end, :]
        
        try:
            xi_input  = x_input[line]
            arg_input = np.where(xi_input[0] == vector[:,0])[0][0]
        except:
            raise IndexError("[Error] arg_input was removed during slicing")
        
        """ Column Densities N_+(mu, s) & N_-(mu, s)"""

        Npmu, mu_local_fwd, dmu_fwd, t_fwd = column_density(vector, field, density, "fwd")
        Nmmu, mu_local_bwd, dmu_bwd, t_bwd = column_density(vector[::-1, :], field[::-1], density[::-1], "bwd")

        Nmir_fwd = mirrored_column_density(vector, field, density, Npmu, 'mir_fwd')
        Nmir_bwd = mirrored_column_density(vector[::-1,:], field[::-1], density[::-1], Nmmu, 'mir_bwd')

        """ Ionization Rate for N = N(s) """
        
        zeta_mir_fwd, zeta_mui_mir_fwd, spectra_fwd  = ionization_rate(Nmir_fwd, mu_local_fwd, dmu_fwd, 'mir_fwd', m=m)
        zeta_mir_bwd, zeta_mui_mir_bwd, spectra_bwd  = ionization_rate(Nmir_bwd, mu_local_bwd, dmu_bwd, 'mir_bwd', m=m)

        Nmir = np.sum(Nmir_fwd + Nmir_bwd[::-1], axis=1) # Adding the corresponding 
        zeta = (zeta_mir_fwd+ zeta_mir_bwd[::-1])          

        zeta_at_x[line] = zeta[arg_input]
        nmir_at_x[line] = Nmir[arg_input]
        local_spectra_at_x[line, :] = spectra_fwd[arg_input,:] + spectra_bwd[::-1,:][arg_input,:]

        return nmir_at_x, zeta_at_x, local_spectra_at_x
    
""" Energies """

def eval_reduction(field, numb, follow_index, threshold):

    R10      = []
    Numb100  = []
    B100     = []

    _, m = field.shape

    flag = False
    filter_mask = np.ones(m).astype(bool)
    dead = 0
    for i in range(m):

        mask10 = np.where(numb[:, i] > threshold)[0]
        if mask10.size > 0:
            start, end = mask10[0], mask10[-1]
            if start <= follow_index <= end:
                try:
                    numb10   = numb[start:end+1, i]
                    bfield10 = field[start:end+1, i]
                    p_r = follow_index - start
                    B_r = bfield10[p_r]
                    n_r = numb10[p_r]
                except IndexError:
                    raise ValueError(f"\nTrying to slice beyond bounds for column {i}. "
                                    f"start={start}, end={end}, shape={numb.shape}")
            else:
                print(f"\n[Info] follow_index {follow_index} outside threshold interval for column {i}.")
                if follow_index >= numb.shape[0]:
                    raise ValueError(f"follow_index {follow_index} is out of bounds for shape {numb.shape}")
                numb10   = np.array([numb[follow_index, i]])
                bfield10 = np.array([field[follow_index, i]])
                p_r = 0
                B_r = bfield10[p_r]
                n_r = numb10[p_r]
        else:
            print(f"\n[Info] No densities > {threshold} cm-3 found for column {i}. Using follow_index fallback.")
            if follow_index >= numb.shape[0]:
                raise ValueError(f"\nfollow_index {follow_index} is out of bounds for shape {numb.shape}")
            numb10   = np.array([numb[follow_index, i]])
            bfield10 = np.array([field[follow_index, i]])
            p_r = 0
            B_r = bfield10[p_r]
            n_r = numb10[p_r]

        #print("p_r: ", p_r)
        if not (0 <= p_r < bfield10.shape[0]):
            raise IndexError(f"\np_r={p_r} is out of bounds for bfield10 of length {len(bfield10)}")

        # pockets with density threshold of 10cm-3
        pocket, global_info = pocket_finder(bfield10, numb10, p_r, plot=flag)
        index_pocket, field_pocket = pocket[0], pocket[1]
        flag = False
        p_i = np.searchsorted(index_pocket, p_r)
        from collections import Counter
        most_common_value, count = Counter(bfield10.ravel()) .most_common(1)[0]
    
        if count > 20:
            R = 1.
            #print(f"Most common value: {most_common_value} (appears {count} times): R = ", R)
            R10.append(R)
            Numb100.append(n_r)
            B100.append(B_r)   
            flag = True
            filter_mask[i] = False
            dead +=1
            continue     

        try:
            closest_values = index_pocket[max(0, p_i - 1): min(len(index_pocket), p_i + 1)]
            B_l = min([bfield10[closest_values[0]], bfield10[closest_values[1]]])
            B_h = max([bfield10[closest_values[0]], bfield10[closest_values[1]]])
            # YES! 
            success = True  
        except:
            # NO :c
            R = 1.
            R10.append(R)
            Numb100.append(n_r)
            B100.append(B_r)
            success = False 
            continue

        if success:
            # Ok, our point is between local maxima, is inside a pocket?
            if B_r / B_l < 1:
                # double YES!
                R = 1. - np.sqrt(1 - B_r / B_l)
                R10.append(R)
                Numb100.append(n_r)
                B100.append(B_r)
            else:
                # NO!
                R = 1.
                R10.append(R)
                Numb100.append(n_r)
                B100.append(B_r)

    filter_mask = filter_mask.astype(bool)

    return np.array(R10), np.array(Numb100), np.array(B100), filter_mask

def pocket_finder(bfield, numb, p_r, plot=False):
    #pocket_finder(bfield, p_r, B_r, img=i, plot=False)
    """  
    Finds peaks in a given magnetic field array.

    Args:
        bfield (array-like): Array or list of magnetic field magnitudes.
        cycle (int, optional): Cycle number for saving the plot. Defaults to 0.
        plot (bool, optional): Whether to plot the results. Defaults to False.

    Returns:
        tuple: Contains two tuples:
            - (indexes, peaks): Lists of peak indexes and corresponding peak values.
            - (index_global_max, upline): Indices and value of the global maximum.
    """
    bfield = np.array(bfield)  # Ensure input is a numpy array

    baseline = np.min(bfield)
    upline = np.max(bfield)
    index_global_max = np.where(bfield == upline)[0]
    try:
        idx = index_global_max[0]
    except:
        idx = index_global_max
    upline == bfield[idx]
    ijk = np.argmax(bfield)
    bfield[ijk] = bfield[ijk]*1.001 # if global_max is found in flat region, choose one and scale it 0.001


    # Find left peaks
    Bi = 0.0
    lindex = []
    lpeaks = []
    for i, Bj in enumerate(bfield):
        if Bj < Bi and (len(lpeaks) == 0 or Bi > lpeaks[-1]):  # if True, then we have a peak
            lindex.append(i - 1)
            lpeaks.append(Bi)
        Bi = Bj

    # Find right peaks
    Bi = 0.0
    rindex = []
    rpeaks = []
    for i, Bj in enumerate(reversed(bfield)):
        if Bj < Bi and (len(rpeaks) == 0 or Bi > rpeaks[-1]):  # if True, then we have a peak
            rindex.append(len(bfield) - i)
            rpeaks.append(Bi)
        Bi = Bj

    peaks = lpeaks +  list(reversed(rpeaks))[1:]
    indexes = lindex + list(reversed(rindex))[1:]

    if plot:
        # Find threshold crossing points for 100 cm^-3
        mask = np.log10(numb) < 2  # log10(100) = 2
        slicebelow = mask[:p_r]
        sliceabove = mask[p_r:]
        peaks = np.array(peaks)
        indexes = np.array(indexes)

        try:
            above100 = np.where(sliceabove)[0][0] + p_r
        except IndexError:
            above100 = None

        try:
            below100 = np.where(slicebelow)[0][-1]
        except IndexError:
            below100 = None


    return (indexes, peaks), (index_global_max, upline)

def smooth_pocket_finder(bfield, cycle=0, plot=False):
    """  
    Finds peaks in a given magnetic field array.

    Args:
        bfield (array-like): Array or list of magnetic field magnitudes.
        cycle (int, optional): Cycle number for saving the plot. Defaults to 0.
        plot (bool, optional): Whether to plot the results. Defaults to False.

    Returns:
        tuple: Contains two tuples:
            - (indexes, peaks): Lists of peak indexes and corresponding peak values.
            - (index_global_max, upline): Indices and value of the global maximum.
    """
    bfield = np.array(bfield)  # Ensure input is a numpy array

    baseline = np.min(bfield)
    upline = np.max(bfield)
    index_global_max = np.where(bfield == upline)[0]
    upline == bfield[index_global_max]
    ijk = np.argmax(bfield)
    bfield[ijk] = bfield[ijk]*1.001 # if global_max is found in flat region, choose one and scale it 0.001

    # Find left peaks
    Bi = 0.0
    lindex = []
    lpeaks = []
    for i, Bj in enumerate(bfield):
        if Bj < Bi and (len(lpeaks) == 0 or Bi > lpeaks[-1]):  # if True, then we have a peak
            lindex.append(i - 1)
            lpeaks.append(Bi)
        Bi = Bj

    # Find right peaks
    Bi = 0.0
    rindex = []
    rpeaks = []
    for i, Bj in enumerate(reversed(bfield)):
        if Bj < Bi and (len(rpeaks) == 0 or Bi > rpeaks[-1]):  # if True, then we have a peak
            rindex.append(len(bfield) - i)
            rpeaks.append(Bi)
        Bi = Bj

    peaks = lpeaks +  list(reversed(rpeaks))[1:]
    indexes = lindex + list(reversed(rindex))[1:]
    
    if plot:
        # Create a figure and axes for the subplot layout
        fig, axs = plt.subplots(1, 1, figsize=(8, 6))

        axs.plot(bfield)
        axs.plot(indexes, peaks, "x", color="green")
        axs.plot(indexes, peaks, ":", color="green")
        
        axs.plot(np.ones_like(bfield) * baseline, "--", color="gray")
        axs.set_xlabel("Index")
        axs.set_ylabel("Field")
        axs.set_title("Actual Field Shape")
        axs.legend(["bfield", "all peaks", "index_global_max", "baseline"])
        axs.grid(True)

        # Adjust layout to prevent overlap
        plt.tight_layout()
        # Save the figure
        plt.savefig(f"./field_shape{cycle}.png")
        plt.close(fig)

    return (indexes, peaks), (index_global_max, upline)

def find_insertion_point(array, val):

    for i in range(len(array)):
        if val < array[i]:
            return i  # Insert before index i
    return len(array)  # Insert at the end if p_r is greater than or equal to all elements

def use_lock_and_save(path):
    from filelock import FileLock
    """
    Time ──────────────────────────────▶

    Script 1:  ──[Acquire Lock]─────[Create/Append HDF5]─────[Release Lock]───

    Script 2:             ──[Wait for Lock]─────────────[Append HDF5]─────[Release Lock]───
    """
    lock = FileLock(path + ".lock")

    with lock:  # acquire lock (waits if another process is holding it)

        pass
        # Use this to create and update a dataframe

""" Decorators 
Obtained from: https://towardsdatascience.com/python-decorators-for-data-science-6913f717669a/

I like this for runs that take time and are not in parallel
"""

def timing_decorator(func):
    import time
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} took {end_time - start_time} seconds to run.")
        return result
    return wrapper

import smtplib
import traceback
from email.mime.text import MIMEText

def email_on_failure(sender_email, password, recipient_email):
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # format the error message and traceback
                err_msg = f"Error: {str(e)}nnTraceback:n{traceback.format_exc()}"

                # create the email message
                message = MIMEText(err_msg)
                message['Subject'] = f"{func.__name__} failed"
                message['From'] = sender_email
                message['To'] = recipient_email

                # send the email
                with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
                    smtp.login(sender_email, password)
                    smtp.sendmail(sender_email, recipient_email, message.as_string())
                # re-raise the exception
                raise
        return wrapper
    return decorator

def plot_w_text():
    import matplotlib.gridspec as gridspec

    # Your parameters as a dict
    params = {
        "t total": 1.1949709,
        "E kinet": 0.093147244,
        "E poten": -0.093147244,
        "E total": 1.0,
        "r coord": 12.0,
        "longd":   0.0,
        "a SpinP": 0.7,
        "v local": 0.40393011,
    }

    fig = plt.figure(figsize=(12, 5))
    gs  = gridspec.GridSpec(1, 2, width_ratios=[1.2, 1])  # plot wider than param panel

    # --- Left: your plot ---
    ax_plot = fig.add_subplot(gs[0])
    ax_plot.plot([1, 2, 3], [1, 4, 9])   # replace with your actual plot
    ax_plot.set_title("My Plot")

    # --- Right: parameter panel ---
    ax_text = fig.add_subplot(gs[1])
    ax_text.axis('off')                   # hide axes entirely

    # Build text from params
    lines = [f"{k:<14} = {v}" for k, v in params.items()]
    text  = "\n".join(lines)

    ax_text.text(
        0.05, 0.95,          # position (axes fraction)
        text,
        transform=ax_text.transAxes,
        fontsize=10,
        verticalalignment='top',
        fontfamily='monospace',  # monospace keeps = signs aligned
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.5)
    )

    plt.tight_layout()
    plt.savefig('plot_with_params.png', dpi=150)
    plt.show()


def hull_from_points(points = np.random.rand(100, 2)):

    from scipy.spatial import ConvexHull
    hull = ConvexHull(points)

    fig, ax = plt.subplots()
    ax.scatter(points[:, 0], points[:, 1], color='steelblue')

    # hull.simplices contains the indices of the edges
    for simplex in hull.simplices:
        ax.plot(points[simplex, 0], points[simplex, 1], 'k-')

    plt.show()
    return hull.simplices