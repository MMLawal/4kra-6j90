#!/usr/bin/env python3

import os
import glob
import csv
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

# analysis settings
ANALYSIS_STRIDE = 10

# cylindrical channel definition
PORE_RADIUS = 6.0      # inner pore cylinder for crossing detection
CHANNEL_ZHALF = 15.0   # half-length of channel cylinder for tracking
Z_SIDE = 10.0          # reservoir threshold relative to pore center

# occupancy cylinder
OCC_RADIUS = 6.0
OCC_ZHALF = 12.0

# selections
WATER_SEL = "(resname WAT TIP3 HOH) and name O"
ION_SELECTIONS = {
    "Na+":  "resname Na+",
    "Cl-":  "resname Cl-"
}

############################################
# PREPARE TRAJECTORY
############################################

def prepare_trajectory(u):
    protein = u.select_atoms("protein")
    nonprotein = u.select_atoms("not protein")

    transforms = [
        unwrap(protein),
        center_in_box(protein, center="geometry", wrap=False),
        wrap(nonprotein, compound="residues")
    ]
    u.trajectory.add_transformations(*transforms)
    return u


############################################
# CHANNEL RADIUS
############################################

def compute_channel_radius(universe, step=10):
    print("Running HOLE radius analysis...")

    H = HoleAnalysis(
        universe,
        select="protein",
        cpoint="center_of_geometry",
        cvect=[0, 0, 1],
        executable=HOLE_EXEC
    )

    H.run(step=step)
    profiles = H.results.profiles

    z_all = []
    r_all = []

    for p in profiles.values():
        z_all.append(p.rxn_coord)
        r_all.append(p.radius)

    z_min = max(z.min() for z in z_all)
    z_max = min(z.max() for z in z_all)
    z_common = np.linspace(z_min, z_max, 200)

    r_interp = [np.interp(z_common, z, r) for z, r in zip(z_all, r_all)]
    r_interp = np.array(r_interp)

    r_mean = np.mean(r_interp, axis=0)
    r_std = np.std(r_interp, axis=0)

    constriction_idx = np.argmin(r_mean)
    constriction_z = z_common[constriction_idx]
    constriction_r = r_mean[constriction_idx]

    return z_common, r_mean, r_std, constriction_z, constriction_r


############################################
# GENERIC CHANNEL TRANSPORT ANALYSIS
############################################

def classify_region(zrel, rxy, pore_radius=PORE_RADIUS, z_side=Z_SIDE, channel_zhalf=CHANNEL_ZHALF):
    """
    Region labels:
      -1 : bottom reservoir
       0 : channel interior
      +1 : top reservoir
     None: outside channel capture zone
    """
    if abs(zrel) > channel_zhalf or rxy > pore_radius:
        return None
    if zrel < -z_side:
        return -1
    elif zrel > z_side:
        return +1
    else:
        return 0


def channel_transport(
    universe,
    atom_selection,
    label,
    stride=ANALYSIS_STRIDE,
    pore_radius=PORE_RADIUS,
    channel_zhalf=CHANNEL_ZHALF,
    z_side=Z_SIDE,
    occ_radius=OCC_RADIUS,
    occ_zhalf=OCC_ZHALF
):
    protein = universe.select_atoms("protein")
    atoms = universe.select_atoms(atom_selection)

    if len(atoms) == 0:
        print(f"WARNING: No atoms found for {label} with selection: {atom_selection}")
        return {
            "label": label,
            "n_atoms": 0,
            "time_ns": np.array([]),
            "occupancy": np.array([]),
            "cumulative_events": np.array([]),
            "top_to_bottom": 0,
            "bottom_to_top": 0,
            "total_events": 0,
            "net_flux": 0,
            "mean_occupancy": 0.0
        }

    print(f"Analyzing transport for {label} ({len(atoms)} atoms)")

    # state per atom index
    # entered_from: initial reservoir side for current attempt
    # in_channel: whether particle has entered channel interior since entry
    track = {}

    time_series = []
    occupancy_series = []
    cumulative_events = []

    top_to_bottom = 0
    bottom_to_top = 0

    for ts in universe.trajectory[::stride]:
        center = protein.center_of_geometry()
        x0, y0, z0 = center

        pos = atoms.positions
        dx = pos[:, 0] - x0
        dy = pos[:, 1] - y0
        dz = pos[:, 2] - z0
        rxy = np.sqrt(dx**2 + dy**2)

        # occupancy inside broader analysis cylinder
        in_occ = (rxy <= occ_radius) & (np.abs(dz) <= occ_zhalf)
        occupancy_series.append(np.sum(in_occ))
        time_series.append(ts.time / 1000.0)

        for atom, rr, zz in zip(atoms, rxy, dz):
            pid = atom.index
            region = classify_region(
                zrel=zz,
                rxy=rr,
                pore_radius=pore_radius,
                z_side=z_side,
                channel_zhalf=channel_zhalf
            )

            if pid not in track:
                track[pid] = {
                    "entered_from": None,
                    "in_channel": False,
                    "last_region": None
                }

            state = track[pid]

            # particle outside channel capture zone: keep state but do not update
            # this avoids destroying a potential crossing due to transient wrapping/noise
            if region is None:
                state["last_region"] = None
                continue

            # initialize entry from reservoir
            if state["entered_from"] is None and region in (-1, +1):
                state["entered_from"] = region
                state["in_channel"] = False

            # mark that particle has traversed channel interior
            elif state["entered_from"] is not None and region == 0:
                state["in_channel"] = True

            # count completed crossing only if particle came from one reservoir,
            # passed through channel interior, and reached the opposite reservoir
            elif (
                state["entered_from"] is not None
                and state["in_channel"]
                and region in (-1, +1)
                and region == -state["entered_from"]
            ):
                if state["entered_from"] == +1 and region == -1:
                    top_to_bottom += 1
                elif state["entered_from"] == -1 and region == +1:
                    bottom_to_top += 1

                # start a new possible event from current side
                state["entered_from"] = region
                state["in_channel"] = False

            # if particle returns to its original side after entering channel, reset
            elif (
                state["entered_from"] is not None
                and state["in_channel"]
                and region == state["entered_from"]
            ):
                state["entered_from"] = region
                state["in_channel"] = False

            state["last_region"] = region

        cumulative_events.append(top_to_bottom + bottom_to_top)

    total_events = top_to_bottom + bottom_to_top
    net_flux = bottom_to_top - top_to_bottom

    return {
        "label": label,
        "n_atoms": len(atoms),
        "time_ns": np.array(time_series),
        "occupancy": np.array(occupancy_series),
        "cumulative_events": np.array(cumulative_events),
        "top_to_bottom": top_to_bottom,
        "bottom_to_top": bottom_to_top,
        "total_events": total_events,
        "net_flux": net_flux,
        "mean_occupancy": float(np.mean(occupancy_series)) if occupancy_series else 0.0
    }


############################################
# OUTPUT HELPERS
############################################

def save_transport_summary_csv(all_results, outfile):
    with open(outfile, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "System", "Species", "N_atoms",
            "Top_to_Bottom", "Bottom_to_Top",
            "Total_events", "Net_flux",
            "Mean_occupancy"
        ])
        for system_label, species_dict in all_results.items():
            for species, res in species_dict.items():
                writer.writerow([
                    system_label,
                    species,
                    res["n_atoms"],
                    res["top_to_bottom"],
                    res["bottom_to_top"],
                    res["total_events"],
                    res["net_flux"],
                    f"{res['mean_occupancy']:.3f}"
                ])


def save_timeseries_csv(all_results, outfile):
    with open(outfile, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "System", "Species", "Time_ns",
            "Occupancy", "Cumulative_events"
        ])
        for system_label, species_dict in all_results.items():
            for species, res in species_dict.items():
                for t, occ, ce in zip(res["time_ns"], res["occupancy"], res["cumulative_events"]):
                    writer.writerow([system_label, species, t, occ, ce])


############################################
# MAIN LOOP
############################################

radius_profiles = {}
transport_results = {}

for label, folder in systems.items():
    print("\n==============================")
    print("Analyzing", label)
    print("==============================")

    path = os.path.join(BASE_DIR, folder)
    top = os.path.join(path, "step3_input.parm7")
    traj = sorted(glob.glob(os.path.join(path, "step*_production.dcd")))

    if not os.path.exists(top):
        raise FileNotFoundError(f"Topology not found: {top}")
    if len(traj) == 0:
        raise FileNotFoundError(f"No trajectories found in {path}")

    u = mda.Universe(top, traj)
    u = prepare_trajectory(u)

    # HOLE radius
    z, r, rs, zc, rc = compute_channel_radius(u, step=ANALYSIS_STRIDE)
    radius_profiles[label] = (z, r, rs, zc, rc)

    # transport analyses
    transport_results[label] = {}

    transport_results[label]["Water"] = channel_transport(
        universe=u,
        atom_selection=WATER_SEL,
        label="Water"
    )

    for ion_label, ion_sel in ION_SELECTIONS.items():
        transport_results[label][ion_label] = channel_transport(
            universe=u,
            atom_selection=ion_sel,
            label=ion_label
        )


############################################
# PLOT CHANNEL RADIUS
############################################

plt.figure(figsize=(7, 5))
for label, (z, r, rs, zc, rc) in radius_profiles.items():
    plt.plot(z, r, label=label)
    plt.fill_between(z, r - rs, r + rs, alpha=0.2)

plt.xlabel("Channel Coordinate (Å)")
plt.ylabel("Radius (Å)")
plt.title("OmpF Channel Radius Profile")
plt.legend()
plt.tight_layout()
plt.savefig("channel_radius_profile.png", dpi=300)
plt.close()


############################################
# PLOT CUMULATIVE FLUX BY SPECIES
############################################

species_to_plot = ["Water", "Na+", "Cl-"]

for species in species_to_plot:
    plt.figure(figsize=(7, 5))
    for system_label in systems.keys():
        res = transport_results[system_label][species]
        if len(res["time_ns"]) > 0:
            plt.plot(res["time_ns"], res["cumulative_events"], label=system_label)
    plt.xlabel("Time (ns)")
    plt.ylabel(f"Cumulative {species} Crossing Events")
    plt.title(f"Cumulative {species} Flux Through OmpF")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{species.replace('+','plus').replace('-','minus')}_cumulative_flux.png", dpi=300)
    plt.close()


############################################
# PLOT OCCUPANCY BY SPECIES
############################################

for species in species_to_plot:
    plt.figure(figsize=(7, 5))
    for system_label in systems.keys():
        res = transport_results[system_label][species]
        if len(res["time_ns"]) > 0:
            plt.plot(res["time_ns"], res["occupancy"], label=system_label)
    plt.xlabel("Time (ns)")
    plt.ylabel(f"{species} Count in Channel")
    plt.title(f"Channel {species} Occupancy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{species.replace('+','plus').replace('-','minus')}_channel_occupancy.png", dpi=300)
    plt.close()


############################################
# BAR PLOT TOTAL EVENTS
############################################

for species in species_to_plot:
    labels = list(systems.keys())
    values = [transport_results[k][species]["total_events"] for k in labels]

    plt.figure(figsize=(6, 5))
    plt.bar(labels, values)
    plt.ylabel(f"Complete {species} Crossing Events")
    plt.title(f"{species} Permeation Through OmpF")
    plt.tight_layout()
    plt.savefig(f"{species.replace('+','plus').replace('-','minus')}_permeation_totals.png", dpi=300)
    plt.close()


############################################
# SAVE CSV OUTPUTS
############################################

save_transport_summary_csv(transport_results, "analysis2_transport_summary.csv")
save_timeseries_csv(transport_results, "analysis2_transport_timeseries.csv")


############################################
# PRINT SUMMARY
############################################

print("\n=== Analysis 2 Summary ===")
for system_label, species_dict in transport_results.items():
    print(f"\nSystem: {system_label}")
    z, r, rs, zc, rc = radius_profiles[system_label]
    print(f"  HOLE constriction: z = {zc:.2f} Å, radius = {rc:.2f} Å")

    for species, res in species_dict.items():
        print(
            f"  {species:>5s} | atoms={res['n_atoms']:4d} | "
            f"T->B={res['top_to_bottom']:4d} | B->T={res['bottom_to_top']:4d} | "
            f"Total={res['total_events']:4d} | Net={res['net_flux']:4d} | "
            f"Mean occ={res['mean_occupancy']:.2f}"
        )
