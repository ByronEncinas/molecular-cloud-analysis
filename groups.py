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



from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import numpy as np

# ── colours / styles per case ────────────────────────────────────────────────
CASE_COLOR  = {"ideal": "steelblue", "amb": "darkorange"}
CASE_LABEL  = {"ideal": "ideal MHD", "amb": "+ ambipolar diffusion"}
MARKER_INIT = "o"
MARKER_LAST = "X"
MARKER_SIZE = 60

# ── load data once ────────────────────────────────────────────────────────────
data = {case: pd.read_csv(f'./util/{case}_clouds.csv') for case in cases}

# ── initial position of each cloud ───────────────────────────────────────────
def cloud_positions(df, which, cols=("c_coord_X", "c_coord_Y", "c_coord_Z")):
    """Initial (first) or final (last) position of each cloud."""
    agg = "first" if which == "initial" else "last"
    return (df.groupby("cloud")[list(cols)]
              .agg(agg)
              .rename_axis("cloud")
              .reset_index())

ref_case   = cases[0]   # "ideal"
other_case = cases[1]   # "amb"

COORD_COLS = ["c_coord_X", "c_coord_Y", "c_coord_Z"]

# ── manually specified pairs (ref_cloud, other_cloud) ────────────────────────
manual_pairs = [
    ("cloud-0", "cloud-0"),
    ("cloud-1", "cloud-2"),
    ("cloud-2", "cloud-4"),
    ("cloud-3", "cloud-1"),
    ("cloud-4", "cloud-5"),
    ("cloud-5", "cloud-3"),
]

COORD_COLS = ["c_coord_X", "c_coord_Y", "c_coord_Z"]

pairs = []
for cn_ref, cn_other in manual_pairs:
    pts_ref   = data[ref_case  ][data[ref_case  ]["cloud"] == cn_ref  ][COORD_COLS].values
    pts_other = data[other_case][data[other_case]["cloud"] == cn_other][COORD_COLS].values
    n = min(len(pts_ref), len(pts_other))
    dist = np.linalg.norm(pts_ref[:n] - pts_other[:n], axis=1).mean()
    pairs.append((cn_ref, cn_other, dist))

dists = np.array([p[2] for p in pairs])
threshold = dists.mean() + dists.std()

pairs_filtered = [p for p in pairs if p[2] <= threshold]
pairs_removed  = [p for p in pairs if p[2] >  threshold]

print(f"Distance threshold: {threshold:.2f} pc  (mean={dists.mean():.2f}, std={dists.std():.2f})")
print(f"Kept   ({len(pairs_filtered)}): ", [(p[0], p[1], f"{p[2]:.2f}") for p in pairs_filtered])
print(f"Removed({len(pairs_removed )}): ", [(p[0], p[1], f"{p[2]:.2f}") for p in pairs_removed])

pairs = pairs_filtered

# ── dump matching results to CSV ──────────────────────────────────────────────
df_pairs = pd.DataFrame({
    f"{ref_case}_cloud":   [p[0] for p in pairs],
    f"{other_case}_cloud": [p[1] for p in pairs],
    "mean_dist_pc":        [p[2] for p in pairs],
    "status": "kept"
})
df_removed = pd.DataFrame({
    f"{ref_case}_cloud":   [p[0] for p in pairs_removed],
    f"{other_case}_cloud": [p[1] for p in pairs_removed],
    "mean_dist_pc":        [p[2] for p in pairs_removed],
    "status": "removed"
})
pd.concat([df_pairs, df_removed], ignore_index=True).to_csv(
    "./util/cloud_matches.csv", index=False
)

for ref_name, other_name, dist in pairs:
    print(f"  {ref_case}:{ref_name:>10}  ↔  {other_case}:{other_name:<10}  dist={dist:.2f} pc")

# ── build mosaic layout (3 rows × 2 cols) ────────────────────────────────────
panel_keys = [f"cloud_{i}" for i in range(len(pairs))]
n_cols = 2
n_rows = int(np.ceil(len(pairs) / n_cols))
# pad to fill the grid if odd number of pairs
keys_padded = panel_keys + ["."] * (n_rows * n_cols - len(pairs))
layout = [keys_padded[r*n_cols:(r+1)*n_cols] for r in range(n_rows)]

fig, ax = plt.subplot_mosaic(
    layout,
    figsize=(10, 12),
    gridspec_kw={"wspace": 0.35, "hspace": 0.55},
)

# ── fill each panel ───────────────────────────────────────────────────────────
cloud_name_map = {ref_case:   [p[0] for p in pairs],
                  other_case: [p[1] for p in pairs]}

for i, key in enumerate(panel_keys):
    _ax = ax[key]

    for case in cases:
        cloud_name = cloud_name_map[case][i]
        group = data[case][data[case]["cloud"] == cloud_name]

        idx_init = group.index[0]
        idx_last = group.index[-1]
        c = CASE_COLOR[case]

        _ax.plot(group["c_coord_X"], group["c_coord_Y"],
                 "-", linewidth=2, color=c, alpha=0.7, label=CASE_LABEL[case])
        _ax.scatter(group["c_coord_X"][idx_init], group["c_coord_Y"][idx_init],
                    marker=MARKER_INIT, s=MARKER_SIZE, color=c, zorder=5)
        _ax.scatter(group["c_coord_X"][idx_last], group["c_coord_Y"][idx_last],
                    marker=MARKER_LAST, s=MARKER_SIZE, color=c, zorder=5)

    ref_name   = cloud_name_map[ref_case][i]
    other_name = cloud_name_map[other_case][i]
    dist       = pairs[i][2]
    _ax.set_title(f"{ref_case}:{ref_name} / {other_case}:{other_name}\n"
                  f"initial dist = {dist:.1f} pc",
                  fontsize=FONTSIZE - 4)
    _ax.set_xlabel("X [Pc]", fontsize=FONTSIZE - 4)
    _ax.set_ylabel("Y [Pc]", fontsize=FONTSIZE - 4)
    _ax.set_ylim(150, 250)
    _ax.grid(True, which="both", alpha=ALPHA)
    _ax.tick_params(labelsize=FONTSIZE - 6)

# ── shared legend ─────────────────────────────────────────────────────────────
handles, labels = ax[panel_keys[0]].get_legend_handles_labels()
by_label = dict(zip(labels, handles))
fig.legend(by_label.values(), by_label.keys(),
           loc="lower center", ncol=2,
           fontsize=FONTSIZE - 4, frameon=False,
           bbox_to_anchor=(0.5, -0.02))

plt.savefig("./paper/comparehisclouds.png", dpi=150, bbox_inches="tight")
plt.close(fig)

exit()
# ── colours / styles per case ────────────────────────────────────────────────
CASE_COLOR  = {"ideal": "steelblue", "amb": "darkorange"}
CASE_LABEL  = {"ideal": "ideal MHD", "amb": "+ ambipolar diffusion"}
MARKER_INIT = "o"   # start of trajectory
MARKER_LAST = "X"   # end   of trajectory
MARKER_SIZE = 60

# ── load data once ────────────────────────────────────────────────────────────
data = {case: pd.read_csv(f'./util/{case}_clouds.csv') for case in cases}

# cloud names are taken from the first case; assumed identical across cases
cloud_names = sorted(data[cases[0]]["cloud"].unique())
assert len(cloud_names) == 6, f"Expected 6 clouds, got {len(cloud_names)}"

# ── build mosaic layout (3 rows × 2 cols) ────────────────────────────────────
layout = [cloud_names[0:2],
          cloud_names[2:4],
          cloud_names[4:6]]

fig, ax = plt.subplot_mosaic(
    layout,
    figsize=(10, 12),
    gridspec_kw={"wspace": 0.35, "hspace": 0.55},
)

# ── fill each panel ───────────────────────────────────────────────────────────
for cloud_name in cloud_names:
    _ax = ax[cloud_name]

    for case in cases:
        df    = data[case]
        group = df[df["cloud"] == cloud_name]

        idx_init = group.index[0]
        idx_last = group.index[-1]

        c = CASE_COLOR[case]

        # trajectory line
        _ax.plot(group["c_coord_X"], group["c_coord_Y"],
                 "-", linewidth=2, color=c, alpha=0.7, label=CASE_LABEL[case])

        # start / end markers (same colour, different shapes)
        _ax.scatter(group["c_coord_X"][idx_init], group["c_coord_Y"][idx_init],
                    marker=MARKER_INIT, s=MARKER_SIZE, color=c, zorder=5)
        _ax.scatter(group["c_coord_X"][idx_last], group["c_coord_Y"][idx_last],
                    marker=MARKER_LAST, s=MARKER_SIZE, color=c, zorder=5)

    _ax.set_title(cloud_name, fontsize=FONTSIZE - 2)
    _ax.set_xlabel("X [Pc]", fontsize=FONTSIZE - 4)
    _ax.set_ylabel("Y [Pc]", fontsize=FONTSIZE - 4)
    _ax.set_ylim(150, 250)
    _ax.grid(True, which="both", alpha=ALPHA)
    _ax.tick_params(labelsize=FONTSIZE - 6)

# ── shared legend (one entry per case, placed outside the grid) ───────────────
handles, labels = ax[cloud_names[0]].get_legend_handles_labels()
# deduplicate (each case plotted once per panel → 2 handles)
by_label = dict(zip(labels, handles))
fig.legend(by_label.values(), by_label.keys(),
           loc="lower center", ncol=2,
           fontsize=FONTSIZE - 4, frameon=False,
           bbox_to_anchor=(0.5, -0.02))

plt.savefig("./paper/groupthisclouds.png", dpi=150, bbox_inches="tight")
plt.close(fig)


from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import numpy as np

# ── colours / styles per case ────────────────────────────────────────────────
CASE_COLOR  = {"ideal": "steelblue", "amb": "darkorange"}
CASE_LABEL  = {"ideal": "ideal MHD", "amb": "+ ambipolar diffusion"}
MARKER_INIT = "o"
MARKER_LAST = "X"
MARKER_SIZE = 60

# ── load data once ────────────────────────────────────────────────────────────
data = {case: pd.read_csv(f'./util/{case}_clouds.csv') for case in cases}

# ── compute a representative position for each cloud ─────────────────────────
def cloud_centroids(df, cols=("c_coord_X", "c_coord_Y", "c_coord_Z")):
    """Mean position of each cloud across all timesteps."""
    return (df.groupby("cloud")[list(cols)]
              .mean()
              .rename_axis("cloud")
              .reset_index())

ref_case  = cases[0]   # "ideal"
other_case = cases[1]  # "amb"

cents_ref   = cloud_centroids(data[ref_case])
cents_other = cloud_centroids(data[other_case])

coords_ref   = cents_ref  [["c_coord_X", "c_coord_Y", "c_coord_Z"]].values
coords_other = cents_other[["c_coord_X", "c_coord_Y", "c_coord_Z"]].values

# ── optimal 1-to-1 matching via the Hungarian algorithm ──────────────────────
cost_matrix = cdist(coords_ref, coords_other)          # shape (6, 6)
row_idx, col_idx = linear_sum_assignment(cost_matrix)

# pairs[i] = (ref_cloud_name, other_cloud_name, distance)
pairs = [
    (cents_ref["cloud"].iloc[r],
     cents_other["cloud"].iloc[c],
     cost_matrix[r, c])
    for r, c in zip(row_idx, col_idx)
]

print("Cloud matching (ref → other):")
for ref_name, other_name, dist in pairs:
    print(f"  {ref_case}:{ref_name:>10}  ↔  {other_case}:{other_name:<10}  dist={dist:.2f} pc")

# ── build mosaic layout (3 rows × 2 cols) ────────────────────────────────────
panel_keys = [f"cloud_{i}" for i in range(len(pairs))]
layout = [panel_keys[0:2],
          panel_keys[2:4],
          panel_keys[4:6]]

fig, ax = plt.subplot_mosaic(
    layout,
    figsize=(10, 12),
    gridspec_kw={"wspace": 0.35, "hspace": 0.55},
)

# ── fill each panel ───────────────────────────────────────────────────────────
cloud_name_map = {ref_case: [p[0] for p in pairs],
                  other_case: [p[1] for p in pairs]}

for i, key in enumerate(panel_keys):
    _ax = ax[key]

    for case in cases:
        cloud_name = cloud_name_map[case][i]
        df    = data[case]
        group = df[df["cloud"] == cloud_name]

        idx_init = group.index[0]
        idx_last = group.index[-1]
        c = CASE_COLOR[case]

        _ax.plot(group["c_coord_X"], group["c_coord_Y"],
                 "-", linewidth=2, color=c, alpha=0.7, label=CASE_LABEL[case])
        _ax.scatter(group["c_coord_X"][idx_init], group["c_coord_Y"][idx_init],
                    marker=MARKER_INIT, s=MARKER_SIZE, color=c, zorder=5)
        _ax.scatter(group["c_coord_X"][idx_last], group["c_coord_Y"][idx_last],
                    marker=MARKER_LAST, s=MARKER_SIZE, color=c, zorder=5)

    ref_name   = cloud_name_map[ref_case][i]
    other_name = cloud_name_map[other_case][i]
    dist       = pairs[i][2]
    _ax.set_title(f"{ref_case}:{ref_name} / {other_case}:{other_name}\n"
                  f"centroid dist = {dist:.1f} pc",
                  fontsize=FONTSIZE - 4)
    _ax.set_xlabel("X [Pc]", fontsize=FONTSIZE - 4)
    _ax.set_ylabel("Y [Pc]", fontsize=FONTSIZE - 4)
    _ax.set_ylim(150, 250)
    _ax.grid(True, which="both", alpha=ALPHA)
    _ax.tick_params(labelsize=FONTSIZE - 6)

# ── shared legend ─────────────────────────────────────────────────────────────
handles, labels = ax[panel_keys[0]].get_legend_handles_labels()
by_label = dict(zip(labels, handles))
fig.legend(by_label.values(), by_label.keys(),
           loc="lower center", ncol=2,
           fontsize=FONTSIZE - 4, frameon=False,
           bbox_to_anchor=(0.5, -0.02))

plt.savefig("./paper/comparehisclouds.png", dpi=150, bbox_inches="tight")
plt.close(fig)


from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import numpy as np

# ── colours / styles per case ────────────────────────────────────────────────
CASE_COLOR  = {"ideal": "steelblue", "amb": "darkorange"}
CASE_LABEL  = {"ideal": "ideal MHD", "amb": "+ ambipolar diffusion"}
MARKER_INIT = "o"
MARKER_LAST = "X"
MARKER_SIZE = 60

# ── load data once ────────────────────────────────────────────────────────────
data = {case: pd.read_csv(f'./util/{case}_clouds.csv') for case in cases}

# ── initial position of each cloud ───────────────────────────────────────────
def cloud_initial_positions(df, cols=("c_coord_X", "c_coord_Y", "c_coord_Z")):
    """Position of each cloud at its first recorded timestep."""
    return (df.groupby("cloud")[list(cols)]
              .first()
              .rename_axis("cloud")
              .reset_index())

ref_case   = cases[0]   # "ideal"
other_case = cases[1]   # "amb"

pos_ref   = cloud_initial_positions(data[ref_case])
pos_other = cloud_initial_positions(data[other_case])

coords_ref   = pos_ref  [["c_coord_X", "c_coord_Y", "c_coord_Z"]].values
coords_other = pos_other[["c_coord_X", "c_coord_Y", "c_coord_Z"]].values

# ── optimal 1-to-1 matching: minimises total initial-position distance ────────
cost_matrix = cdist(coords_ref, coords_other)          # shape (6, 6)
row_idx, col_idx = linear_sum_assignment(cost_matrix)  # Hungarian algorithm

pairs = [
    (pos_ref["cloud"].iloc[r],
     pos_other["cloud"].iloc[c],
     cost_matrix[r, c])
    for r, c in zip(row_idx, col_idx)
]

print(f"Cloud matching (minimised total distance = {sum(p[2] for p in pairs):.2f} pc)")
for ref_name, other_name, dist in pairs:
    print(f"  {ref_case}:{ref_name:>10}  ↔  {other_case}:{other_name:<10}  dist={dist:.2f} pc")

# ── build mosaic layout (3 rows × 2 cols) ────────────────────────────────────
panel_keys = [f"cloud_{i}" for i in range(len(pairs))]
layout = [panel_keys[0:2],
          panel_keys[2:4],
          panel_keys[4:6]]

fig, ax = plt.subplot_mosaic(
    layout,
    figsize=(10, 12),
    gridspec_kw={"wspace": 0.35, "hspace": 0.55},
)

# ── fill each panel ───────────────────────────────────────────────────────────
cloud_name_map = {ref_case:   [p[0] for p in pairs],
                  other_case: [p[1] for p in pairs]}

for i, key in enumerate(panel_keys):
    _ax = ax[key]

    for case in cases:
        cloud_name = cloud_name_map[case][i]
        group = data[case][data[case]["cloud"] == cloud_name]

        idx_init = group.index[0]
        idx_last = group.index[-1]
        c = CASE_COLOR[case]

        _ax.plot(group["c_coord_X"], group["c_coord_Y"],
                 "-", linewidth=2, color=c, alpha=0.7, label=CASE_LABEL[case])
        _ax.scatter(group["c_coord_X"][idx_init], group["c_coord_Y"][idx_init],
                    marker=MARKER_INIT, s=MARKER_SIZE, color=c, zorder=5)
        _ax.scatter(group["c_coord_X"][idx_last], group["c_coord_Y"][idx_last],
                    marker=MARKER_LAST, s=MARKER_SIZE, color=c, zorder=5)

    ref_name   = cloud_name_map[ref_case][i]
    other_name = cloud_name_map[other_case][i]
    dist       = pairs[i][2]
    _ax.set_title(f"{ref_case}:{ref_name} / {other_case}:{other_name}\n"
                  f"initial dist = {dist:.1f} pc",
                  fontsize=FONTSIZE - 4)
    _ax.set_xlabel("X [Pc]", fontsize=FONTSIZE - 4)
    _ax.set_ylabel("Y [Pc]", fontsize=FONTSIZE - 4)
    _ax.set_ylim(150, 250)
    _ax.grid(True, which="both", alpha=ALPHA)
    _ax.tick_params(labelsize=FONTSIZE - 6)

# ── shared legend ─────────────────────────────────────────────────────────────
handles, labels = ax[panel_keys[0]].get_legend_handles_labels()
by_label = dict(zip(labels, handles))
fig.legend(by_label.values(), by_label.keys(),
           loc="lower center", ncol=2,
           fontsize=FONTSIZE - 4, frameon=False,
           bbox_to_anchor=(0.5, -0.02))

plt.savefig("./paper/compare2thisclouds.png", dpi=150, bbox_inches="tight")
plt.close(fig)



from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import numpy as np

# ── colours / styles per case ────────────────────────────────────────────────
CASE_COLOR  = {"ideal": "steelblue", "amb": "darkorange"}
CASE_LABEL  = {"ideal": "ideal MHD", "amb": "+ ambipolar diffusion"}
MARKER_INIT = "o"
MARKER_LAST = "X"
MARKER_SIZE = 60

# ── load data once ────────────────────────────────────────────────────────────
data = {case: pd.read_csv(f'./util/{case}_clouds.csv') for case in cases}

# ── initial position of each cloud ───────────────────────────────────────────
def cloud_positions(df, which, cols=("c_coord_X", "c_coord_Y", "c_coord_Z")):
    """Initial (first) or final (last) position of each cloud."""
    agg = "first" if which == "initial" else "last"
    return (df.groupby("cloud")[list(cols)]
              .agg(agg)
              .rename_axis("cloud")
              .reset_index())

ref_case   = cases[0]   # "ideal"
other_case = cases[1]   # "amb"

COORD_COLS = ["c_coord_X", "c_coord_Y", "c_coord_Z"]

pos_init_ref   = cloud_positions(data[ref_case],   "initial")[COORD_COLS].values
pos_init_other = cloud_positions(data[other_case], "initial")[COORD_COLS].values
pos_last_ref   = cloud_positions(data[ref_case],   "final"  )[COORD_COLS].values
pos_last_other = cloud_positions(data[other_case], "final"  )[COORD_COLS].values

# cloud names stay aligned with the grouped order
cloud_names_ref   = cloud_positions(data[ref_case],   "initial")["cloud"]
cloud_names_other = cloud_positions(data[other_case], "initial")["cloud"]

# ── optimal 1-to-1 matching: minimises sum of initial + final distances ───────
cost_matrix = cdist(pos_init_ref, pos_init_other) + cdist(pos_last_ref, pos_last_other)
row_idx, col_idx = linear_sum_assignment(cost_matrix)  # Hungarian algorithm

pairs = [
    (cloud_names_ref.iloc[r],
     cloud_names_other.iloc[c],
     cost_matrix[r, c])
    for r, c in zip(row_idx, col_idx)
]

print(f"Cloud matching (minimised total initial+final distance = {sum(p[2] for p in pairs):.2f} pc)")
for ref_name, other_name, dist in pairs:
    print(f"  {ref_case}:{ref_name:>10}  ↔  {other_case}:{other_name:<10}  dist={dist:.2f} pc")

# ── build mosaic layout (3 rows × 2 cols) ────────────────────────────────────
panel_keys = [f"cloud_{i}" for i in range(len(pairs))]
layout = [panel_keys[0:2],
          panel_keys[2:4],
          panel_keys[4:6]]

fig, ax = plt.subplot_mosaic(
    layout,
    figsize=(10, 12),
    gridspec_kw={"wspace": 0.35, "hspace": 0.55},
)

# ── fill each panel ───────────────────────────────────────────────────────────
cloud_name_map = {ref_case:   [p[0] for p in pairs],
                  other_case: [p[1] for p in pairs]}

for i, key in enumerate(panel_keys):
    _ax = ax[key]

    for case in cases:
        cloud_name = cloud_name_map[case][i]
        group = data[case][data[case]["cloud"] == cloud_name]

        idx_init = group.index[0]
        idx_last = group.index[-1]
        c = CASE_COLOR[case]

        _ax.plot(group["c_coord_X"], group["c_coord_Y"],
                 "-", linewidth=2, color=c, alpha=0.7, label=CASE_LABEL[case])
        _ax.scatter(group["c_coord_X"][idx_init], group["c_coord_Y"][idx_init],
                    marker=MARKER_INIT, s=MARKER_SIZE, color=c, zorder=5)
        _ax.scatter(group["c_coord_X"][idx_last], group["c_coord_Y"][idx_last],
                    marker=MARKER_LAST, s=MARKER_SIZE, color=c, zorder=5)

    ref_name   = cloud_name_map[ref_case][i]
    other_name = cloud_name_map[other_case][i]
    dist       = pairs[i][2]
    _ax.set_title(f"{ref_case}:{ref_name} / {other_case}:{other_name}\n"
                  f"initial dist = {dist:.1f} pc",
                  fontsize=FONTSIZE - 4)
    _ax.set_xlabel("X [Pc]", fontsize=FONTSIZE - 4)
    _ax.set_ylabel("Y [Pc]", fontsize=FONTSIZE - 4)
    _ax.set_ylim(150, 250)
    _ax.grid(True, which="both", alpha=ALPHA)
    _ax.tick_params(labelsize=FONTSIZE - 6)

# ── shared legend ─────────────────────────────────────────────────────────────
handles, labels = ax[panel_keys[0]].get_legend_handles_labels()
by_label = dict(zip(labels, handles))
fig.legend(by_label.values(), by_label.keys(),
           loc="lower center", ncol=2,
           fontsize=FONTSIZE - 4, frameon=False,
           bbox_to_anchor=(0.5, -0.02))

plt.savefig("./paper/compare3thisclouds.png", dpi=150, bbox_inches="tight")
plt.close(fig)

from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import numpy as np

# ── colours / styles per case ────────────────────────────────────────────────
CASE_COLOR  = {"ideal": "steelblue", "amb": "darkorange"}
CASE_LABEL  = {"ideal": "ideal MHD", "amb": "+ ambipolar diffusion"}
MARKER_INIT = "o"
MARKER_LAST = "X"
MARKER_SIZE = 60

# ── load data once ────────────────────────────────────────────────────────────
data = {case: pd.read_csv(f'./util/{case}_clouds.csv') for case in cases}

# ── initial position of each cloud ───────────────────────────────────────────
def cloud_positions(df, which, cols=("c_coord_X", "c_coord_Y", "c_coord_Z")):
    """Initial (first) or final (last) position of each cloud."""
    agg = "first" if which == "initial" else "last"
    return (df.groupby("cloud")[list(cols)]
              .agg(agg)
              .rename_axis("cloud")
              .reset_index())

ref_case   = cases[0]   # "ideal"
other_case = cases[1]   # "amb"

COORD_COLS = ["c_coord_X", "c_coord_Y", "c_coord_Z"]

cloud_names_ref   = sorted(data[ref_case  ]["cloud"].unique())
cloud_names_other = sorted(data[other_case]["cloud"].unique())

# ── optimal 1-to-1 matching: minimises mean pairwise distance across all timesteps ──
cost_matrix = np.zeros((len(cloud_names_ref), len(cloud_names_other)))

for i, cn_ref in enumerate(cloud_names_ref):
    pts_ref = data[ref_case][data[ref_case]["cloud"] == cn_ref][COORD_COLS].values
    for j, cn_other in enumerate(cloud_names_other):
        pts_other = data[other_case][data[other_case]["cloud"] == cn_other][COORD_COLS].values
        # mean of term-by-term distances (truncate to shorter trajectory)
        n = min(len(pts_ref), len(pts_other))
        cost_matrix[i, j] = np.linalg.norm(pts_ref[:n] - pts_other[:n], axis=1).mean()

row_idx, col_idx = linear_sum_assignment(cost_matrix)  # Hungarian algorithm

pairs = [
    (cloud_names_ref[r],
     cloud_names_other[c],
     cost_matrix[r, c])
    for r, c in zip(row_idx, col_idx)
]

print(f"Cloud matching (minimised mean term-by-term distance = {sum(p[2] for p in pairs):.2f} pc)")
for ref_name, other_name, dist in pairs:
    print(f"  {ref_case}:{ref_name:>10}  ↔  {other_case}:{other_name:<10}  dist={dist:.2f} pc")

# ── build mosaic layout (3 rows × 2 cols) ────────────────────────────────────
panel_keys = [f"cloud_{i}" for i in range(len(pairs))]
layout = [panel_keys[0:2],
          panel_keys[2:4],
          panel_keys[4:6]]

fig, ax = plt.subplot_mosaic(
    layout,
    figsize=(10, 12),
    gridspec_kw={"wspace": 0.35, "hspace": 0.55},
)

# ── fill each panel ───────────────────────────────────────────────────────────
cloud_name_map = {ref_case:   [p[0] for p in pairs],
                  other_case: [p[1] for p in pairs]}

for i, key in enumerate(panel_keys):
    _ax = ax[key]

    for case in cases:
        cloud_name = cloud_name_map[case][i]
        group = data[case][data[case]["cloud"] == cloud_name]

        idx_init = group.index[0]
        idx_last = group.index[-1]
        c = CASE_COLOR[case]

        _ax.plot(group["c_coord_X"], group["c_coord_Y"],
                 "-", linewidth=2, color=c, alpha=0.7, label=CASE_LABEL[case])
        _ax.scatter(group["c_coord_X"][idx_init], group["c_coord_Y"][idx_init],
                    marker=MARKER_INIT, s=MARKER_SIZE, color=c, zorder=5)
        _ax.scatter(group["c_coord_X"][idx_last], group["c_coord_Y"][idx_last],
                    marker=MARKER_LAST, s=MARKER_SIZE, color=c, zorder=5)

    ref_name   = cloud_name_map[ref_case][i]
    other_name = cloud_name_map[other_case][i]
    dist       = pairs[i][2]
    _ax.set_title(f"{ref_case}:{ref_name} / {other_case}:{other_name}\n"
                  f"initial dist = {dist:.1f} pc",
                  fontsize=FONTSIZE - 4)
    _ax.set_xlabel("X [Pc]", fontsize=FONTSIZE - 4)
    _ax.set_ylabel("Y [Pc]", fontsize=FONTSIZE - 4)
    _ax.set_ylim(150, 250)
    _ax.grid(True, which="both", alpha=ALPHA)
    _ax.tick_params(labelsize=FONTSIZE - 6)

# ── shared legend ─────────────────────────────────────────────────────────────
handles, labels = ax[panel_keys[0]].get_legend_handles_labels()
by_label = dict(zip(labels, handles))
fig.legend(by_label.values(), by_label.keys(),
           loc="lower center", ncol=2,
           fontsize=FONTSIZE - 4, frameon=False,
           bbox_to_anchor=(0.5, -0.02))

plt.savefig("./paper/trajthisclouds.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ── uses: data, pairs, cloud_name_map, ref_case, other_case from matching block ──
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import numpy as np

# ── colours / styles per case ────────────────────────────────────────────────
CASE_COLOR  = {"ideal": "steelblue", "amb": "darkorange"}
CASE_LABEL  = {"ideal": "ideal MHD", "amb": "+ ambipolar diffusion"}
MARKER_INIT = "o"
MARKER_LAST = "X"
MARKER_SIZE = 60

# ── load data once ────────────────────────────────────────────────────────────
data = {case: pd.read_csv(f'./util/{case}_clouds.csv') for case in cases}

# ── initial position of each cloud ───────────────────────────────────────────
def cloud_positions(df, which, cols=("c_coord_X", "c_coord_Y", "c_coord_Z")):
    """Initial (first) or final (last) position of each cloud."""
    agg = "first" if which == "initial" else "last"
    return (df.groupby("cloud")[list(cols)]
              .agg(agg)
              .rename_axis("cloud")
              .reset_index())

ref_case   = cases[0]   # "ideal"
other_case = cases[1]   # "amb"

COORD_COLS = ["c_coord_X", "c_coord_Y", "c_coord_Z"]

# ── manually specified pairs (ref_cloud, other_cloud) ────────────────────────
manual_pairs = [
    ("cloud-0", "cloud-0"),
    ("cloud-1", "cloud-2"),
    ("cloud-2", "cloud-4"),
    ("cloud-3", "cloud-1"),
    ("cloud-4", "cloud-5"),
    ("cloud-5", "cloud-3"),
]

COORD_COLS = ["c_coord_X", "c_coord_Y", "c_coord_Z"]

pairs = []
for cn_ref, cn_other in manual_pairs:
    pts_ref   = data[ref_case  ][data[ref_case  ]["cloud"] == cn_ref  ][COORD_COLS].values
    pts_other = data[other_case][data[other_case]["cloud"] == cn_other][COORD_COLS].values
    n = min(len(pts_ref), len(pts_other))
    dist = np.linalg.norm(pts_ref[:n] - pts_other[:n], axis=1).mean()
    pairs.append((cn_ref, cn_other, dist))

print("Cloud pairs:")
for ref_name, other_name, dist in pairs:
    print(f"  {ref_case}:{ref_name:>10}  ↔  {other_case}:{other_name:<10}  mean dist={dist:.2f} pc")
for ref_name, other_name, dist in pairs:
    print(f"  {ref_case}:{ref_name:>10}  ↔  {other_case}:{other_name:<10}  dist={dist:.2f} pc")

# ── build mosaic layout (3 rows × 2 cols) ────────────────────────────────────
panel_keys = [f"cloud_{i}" for i in range(len(pairs))]
layout = [panel_keys[0:2],
          panel_keys[2:4],
          panel_keys[4:6]]

fig, ax = plt.subplot_mosaic(
    layout,
    figsize=(10, 12),
    gridspec_kw={"wspace": 0.35, "hspace": 0.55},
)

# ── fill each panel ───────────────────────────────────────────────────────────
cloud_name_map = {ref_case:   [p[0] for p in pairs],
                  other_case: [p[1] for p in pairs]}

for i, key in enumerate(panel_keys):
    _ax = ax[key]

    for case in cases:
        cloud_name = cloud_name_map[case][i]
        group = data[case][data[case]["cloud"] == cloud_name]

        idx_init = group.index[0]
        idx_last = group.index[-1]
        c = CASE_COLOR[case]

        _ax.plot(group["c_coord_X"], group["c_coord_Y"],
                 "-", linewidth=2, color=c, alpha=0.7, label=CASE_LABEL[case])
        _ax.scatter(group["c_coord_X"][idx_init], group["c_coord_Y"][idx_init],
                    marker=MARKER_INIT, s=MARKER_SIZE, color=c, zorder=5)
        _ax.scatter(group["c_coord_X"][idx_last], group["c_coord_Y"][idx_last],
                    marker=MARKER_LAST, s=MARKER_SIZE, color=c, zorder=5)

    ref_name   = cloud_name_map[ref_case][i]
    other_name = cloud_name_map[other_case][i]
    dist       = pairs[i][2]
    _ax.set_title(f"{ref_case}:{ref_name} / {other_case}:{other_name}\n"
                  f"initial dist = {dist:.1f} pc",
                  fontsize=FONTSIZE - 4)
    _ax.set_xlabel("X [Pc]", fontsize=FONTSIZE - 4)
    _ax.set_ylabel("Y [Pc]", fontsize=FONTSIZE - 4)
    _ax.set_ylim(150, 250)
    _ax.grid(True, which="both", alpha=ALPHA)
    _ax.tick_params(labelsize=FONTSIZE - 6)

# ── shared legend ─────────────────────────────────────────────────────────────
handles, labels = ax[panel_keys[0]].get_legend_handles_labels()
by_label = dict(zip(labels, handles))
fig.legend(by_label.values(), by_label.keys(),
           loc="lower center", ncol=2,
           fontsize=FONTSIZE - 4, frameon=False,
           bbox_to_anchor=(0.5, -0.02))

plt.savefig("./paper/pairthisclouds.png", dpi=150, bbox_inches="tight")
plt.close(fig)