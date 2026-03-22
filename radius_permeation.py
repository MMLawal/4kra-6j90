#!/usr/bin/env python3

import os
import glob
import numpy as np
import matplotlib.pyplot as plt

import MDAnalysis as mda
from mdahole2.analysis import HoleAnalysis
from MDAnalysis.transformations import unwrap, center_in_box, wrap

############################################
# SETTINGS
############################################

BASE_DIR = "4kra"

systems = {
    "Mangiferin": "mangi",
    "Ciprofloxacin": "cipro",
    "Azadirachtin": "azad",
    "Apo": "protein"
}

HOLE_EXEC = "/u/mlawal/miniconda3/pkgs/hole2-2.3.1-h3b12eaf_1/bin/hole"

# channel-permeation settings
PERM_STRIDE = 10
PORE_RADIUS = 6.0      # Å, cylindrical pore capture radius
Z_SIDE = 10.0          # Å, defines top vs bottom reservoir relative to pore center
OCC_RADIUS = 6.0       # Å, occupancy cylinder radius
OCC_ZHALF = 12.0       # Å, half-length of channel occupancy region

############################################
# PREPARE TRAJECTORY (AUTO-IMAGE)
############################################

def prepare_trajectory(u):

    protein = u.select_atoms("protein")
    solvent_ions = u.select_atoms("not protein or resname UNL or resname UNK")

    transforms = [
        unwrap(protein),
        center_in_box(protein, center="geometry", wrap=False),
        wrap(solvent_ions, compound="residues")
    ]

    u.trajectory.add_transformations(*transforms)
    return u


############################################
# CHANNEL RADIUS
############################################

def compute_channel_radius(universe):

    print("Running HOLE radius analysis")

    H = HoleAnalysis(
        universe,
        select="protein",
        cpoint="center_of_geometry",
        cvect=[0, 0, 1],
        executable=HOLE_EXEC
    )

    H.run(step=10)

    profiles = H.results.profiles

    z_all = []
    r_all = []

    for p in profiles.values():
        z_all.append(p.rxn_coord)
        r_all.append(p.radius)

    z_min = max(z.min() for z in z_all)
    z_max = min(z.max() for z in z_all)
    z_common = np.linspace(z_min, z_max, 200)

    r_interp = []
    for z, r in zip(z_all, r_all):
        r_interp.append(np.interp(z_common, z, r))

    r_interp = np.array(r_interp)

    r_mean = np.mean(r_interp, axis=0)
    r_std = np.std(r_interp, axis=0)

    constriction_idx = np.argmin(r_mean)
    constriction_z = z_common[constriction_idx]
    constriction_r = r_mean[constriction_idx]

    return z_common, r_mean, r_std, constriction_z, constriction_r


############################################
# WATER PERMEATION THROUGH CYLINDRICAL PORE
############################################

def water_permeation(universe, pore_radius=PORE_RADIUS, z_side=Z_SIDE, stride=PERM_STRIDE):

    protein = universe.select_atoms("protein")
    waters = universe.select_atoms("(resname WAT TIP3 HOH) and name O")

    # track each water molecule through the pore
    track = {}
    cumulative_events = []
    occupancy_series = []
    time_series = []

    n_events = 0

    for ts in universe.trajectory[::stride]:

        # define pore center from protein COG at this frame
        center = protein.center_of_geometry()
        x0, y0, z0 = center

        pos = waters.positions

        dx = pos[:, 0] - x0
        dy = pos[:, 1] - y0
        dz = pos[:, 2] - z0

        rxy = np.sqrt(dx**2 + dy**2)

        # occupancy metric: waters inside channel cylinder
        in_occ = (rxy <= OCC_RADIUS) & (np.abs(dz) <= OCC_ZHALF)
        occupancy_series.append(np.sum(in_occ))

        # permeation metric: only waters inside tighter pore cylinder
        in_pore = (rxy <= pore_radius)

        time_series.append(ts.time / 1000.0)  # ps -> ns

        current_ids = set()

        for atom, rp, zrel in zip(waters, in_pore, dz):

            wid = atom.index

            if not rp:
                # reset if water leaves pore region
                if wid in track:
                    track[wid]["entered_from"] = None
                    track[wid]["last_region"] = None
                continue

            current_ids.add(wid)

            if zrel < -z_side:
                region = -1
            elif zrel > z_side:
                region = +1
            else:
                region = 0

            if wid not in track:
                track[wid] = {"entered_from": None, "last_region": None}

            entered_from = track[wid]["entered_from"]
            last_region = track[wid]["last_region"]

            # water first appears on one side of the pore
            if entered_from is None and region in (-1, +1):
                track[wid]["entered_from"] = region

            # count complete translocation
            elif entered_from is not None and region in (-1, +1):
                if region == -entered_from:
                    n_events += 1
                    # after a completed crossing, reset origin to current side
                    track[wid]["entered_from"] = region

            track[wid]["last_region"] = region

        cumulative_events.append(n_events)

    return {
        "total_events": n_events,
        "time_ns": np.array(time_series),
        "cumulative_events": np.array(cumulative_events),
        "occupancy": np.array(occupancy_series)
    }


############################################
# MAIN LOOP
############################################

radius_profiles = {}
permeation_results = {}

for label, folder in systems.items():

    print("\n==============================")
    print("Analyzing", label)
    print("==============================")

    path = os.path.join(BASE_DIR, folder)

    top = os.path.join(path, "step3_input.parm7")
    traj = sorted(glob.glob(os.path.join(path, "step*_production.dcd")))

    u = mda.Universe(top, traj)
    u = prepare_trajectory(u)

    ################################
    # HOLE radius
    ################################
    z, r, rs, zc, rc = compute_channel_radius(u)
    radius_profiles[label] = (z, r, rs, zc, rc)

    ################################
    # Water permeation
    ################################
    permeation_results[label] = water_permeation(u)


############################################
# PLOT CHANNEL RADIUS
############################################

plt.figure(figsize=(7, 5))

for label, (z, r, rs, zc, rc) in radius_profiles.items():
    plt.plot(z, r, label=label)
    plt.fill_between(z, r-rs, r+rs, alpha=0.2)

plt.xlabel("Channel Coordinate (Å)")
plt.ylabel("Radius (Å)")
plt.title("OmpF Channel Radius Profile")
plt.legend()
plt.tight_layout()
plt.savefig("channel_radius_profile.png", dpi=300)
plt.close()


############################################
# PLOT TOTAL WATER PERMEATION EVENTS
############################################

labels = list(permeation_results.keys())
values = [permeation_results[k]["total_events"] for k in labels]

plt.figure(figsize=(6, 5))
plt.bar(labels, values)
plt.ylabel("Complete Water Permeation Events")
plt.title("Water Permeation Through OmpF")
plt.tight_layout()
plt.savefig("water_permeation_totals.png", dpi=300)
plt.close()


############################################
# PLOT CUMULATIVE WATER FLUX
############################################

plt.figure(figsize=(7, 5))

for label, res in permeation_results.items():
    plt.plot(res["time_ns"], res["cumulative_events"], label=label)

plt.xlabel("Time (ns)")
plt.ylabel("Cumulative Water Permeation Events")
plt.title("Cumulative Water Flux Through OmpF")
plt.legend()
plt.tight_layout()
plt.savefig("water_permeation_cumulative.png", dpi=300)
plt.close()


############################################
# PLOT CHANNEL WATER OCCUPANCY
############################################

plt.figure(figsize=(7, 5))

for label, res in permeation_results.items():
    plt.plot(res["time_ns"], res["occupancy"], label=label)

plt.xlabel("Time (ns)")
plt.ylabel("Waters in Channel Cylinder")
plt.title("Channel Water Occupancy")
plt.legend()
plt.tight_layout()
plt.savefig("water_channel_occupancy.png", dpi=300)
plt.close()


############################################
# PRINT RESULTS
############################################

print("\nWater Permeation Summary")
for k, v in permeation_results.items():
    print(f"{k}: {v['total_events']} complete crossings")
