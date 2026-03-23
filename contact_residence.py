#!/usr/bin/env python3

import os
import glob
import numpy as np
import pandas as pd
import MDAnalysis as mda
from MDAnalysis.lib.distances import distance_array
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ============================================================
# SYSTEM DEFINITIONS
# ============================================================

BASE_DIRS = ("4kra", "dnagyrase")

SYSTEMS = {
    "mangi": "Mangiferin",
    "cipro": "Ciprofloxacin",
    "azad": "Azadirachtin",
    "protein": "Apo",
}

LIGAND_MAP = {
    "4kra": {"protein": "1-341", "ligand": 342},
    "dnagyrase": {"protein": "1-390", "ligand": 391},
}

TOPOLOGY_FILE = "step3_input.parm7"
TRAJECTORY_FILE = "step1_production.dcd"

# ============================================================
# ANALYSIS PARAMETERS
# ============================================================

POCKET_CUTOFF = 5.0      # Å, define pocket residues from first frame
CONTACT_CUTOFF = 5.0     # Å, ligand-residue contact threshold
STRIDE = 10              # analyze every Nth frame
MIN_POCKET_RES = 1       # ligand considered bound if >= this number of pocket residues contact ligand

# ============================================================
# SYSTEM DISCOVERY
# ============================================================

def discover_systems():
    systems = []

    for protein in BASE_DIRS:
        for lig_dir in sorted(glob.glob(os.path.join(protein, "*"))):
            lig_key = os.path.basename(lig_dir)

            if lig_key not in SYSTEMS:
                continue
            if lig_key == "protein":
                continue   # skip apo for ligand residence analysis

            topo = os.path.join(lig_dir, TOPOLOGY_FILE)
            traj = os.path.join(lig_dir, TRAJECTORY_FILE)

            if not (os.path.exists(topo) and os.path.exists(traj)):
                continue

            systems.append({
                "protein": protein,
                "ligand_key": lig_key,
                "ligand_name": SYSTEMS[lig_key],
                "topology": topo,
                "trajectory": traj,
                "ligand_resid": LIGAND_MAP[protein]["ligand"],
                "protein_resids": LIGAND_MAP[protein]["protein"],
            })

    return systems

# ============================================================
# POCKET DEFINITION
# ============================================================

def define_pocket_residues(u, ligand_resid, cutoff=POCKET_CUTOFF):
    """
    Pocket = protein residues within cutoff Å of ligand in first frame.
    """
    ligand = u.select_atoms(f"resid {ligand_resid} and not name H*")
    protein_residues = u.select_atoms("protein and not name H*").residues

    pocket = []

    for res in protein_residues:
        atoms = res.atoms.select_atoms("not name H*")
        if len(atoms) == 0:
            continue

        d = distance_array(ligand.positions, atoms.positions)
        if np.min(d) <= cutoff:
            pocket.append((int(res.resid), str(res.resname)))

    if len(pocket) == 0:
        raise ValueError("No pocket residues found. Increase POCKET_CUTOFF.")

    return pocket

# ============================================================
# CONTACT PERSISTENCE + BOUND STATE
# ============================================================

def analyze_contacts(system):
    u = mda.Universe(system["topology"], system["trajectory"])
    ligand = u.select_atoms(f"resid {system['ligand_resid']} and not name H*")

    if len(ligand) == 0:
        raise ValueError(f"No ligand atoms found for resid {system['ligand_resid']}")

    pocket = define_pocket_residues(u, system["ligand_resid"])
    pocket_resids = [r for r, _ in pocket]

    residue_groups = {
        resid: u.select_atoms(f"protein and resid {resid} and not name H*")
        for resid in pocket_resids
    }

    contact_counts = {resid: 0 for resid in pocket_resids}
    bound_series = []
    time_series = []

    n_frames = 0
    dt_ps = u.trajectory.dt if u.trajectory.dt is not None else 1.0

    for iframe, ts in enumerate(u.trajectory[::STRIDE]):
        n_frames += 1
        time_series.append(ts.time / 1000.0 if ts.time is not None else iframe * dt_ps * STRIDE / 1000.0)

        n_contacts_this_frame = 0

        for resid, group in residue_groups.items():
            d = distance_array(ligand.positions, group.positions)
            if np.min(d) <= CONTACT_CUTOFF:
                contact_counts[resid] += 1
                n_contacts_this_frame += 1

        bound_series.append(1 if n_contacts_this_frame >= MIN_POCKET_RES else 0)

    # residue persistence table
    persistence_rows = []
    pocket_map = dict(pocket)

    for resid in pocket_resids:
        persistence_rows.append({
            "protein": system["protein"],
            "ligand": system["ligand_name"],
            "resid": resid,
            "resname": pocket_map[resid],
            "persistence": contact_counts[resid] / n_frames
        })

    persistence_df = pd.DataFrame(persistence_rows)

    result = {
        "protein": system["protein"],
        "ligand": system["ligand_name"],
        "time_ns": np.array(time_series),
        "bound_series": np.array(bound_series, dtype=int),
        "persistence_df": persistence_df,
        "n_frames": n_frames,
        "pocket_residues": pocket
    }

    return result

# ============================================================
# SURVIVAL PROBABILITY
# ============================================================

def survival_probability(bound_series):
    """
    S(t) = probability that a bound state persists continuously for at least t.
    Computed from contiguous bound segments.
    """
    bound_series = np.asarray(bound_series, dtype=int)

    # lengths of contiguous bound segments
    lengths = []
    current = 0
    for x in bound_series:
        if x == 1:
            current += 1
        else:
            if current > 0:
                lengths.append(current)
            current = 0
    if current > 0:
        lengths.append(current)

    if len(lengths) == 0:
        return np.array([0.0]), np.array([0.0])

    max_len = max(lengths)
    tau = np.arange(1, max_len + 1)

    S = np.array([np.mean(np.array(lengths) >= t) for t in tau], dtype=float)
    return tau, S

def estimate_residence_time(tau_frames, S, dt_ns):
    """
    Residence time proxy = integral of survival probability.
    """
    if len(S) == 0:
        return 0.0
    tau_ns = tau_frames * dt_ns
    return np.trapezoid(S, tau_ns)

# ============================================================
# PLOTTING
# ============================================================

def plot_survival(time_lag_ns, S, outpng, title):
    plt.figure(figsize=(6, 4))
    plt.plot(time_lag_ns, S, lw=2)
    plt.xlabel("Lag time (ns)")
    plt.ylabel("Survival probability")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpng, dpi=300)
    plt.close()

def plot_dissociation(time_lag_ns, S, outpng, title):
    plt.figure(figsize=(6, 4))
    plt.plot(time_lag_ns, 1.0 - S, lw=2)
    plt.xlabel("Lag time (ns)")
    plt.ylabel("Dissociation probability")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpng, dpi=300)
    plt.close()

def plot_persistence_heatmap(df, outpng, title):
    pivot = df.pivot_table(index="resid", columns="ligand", values="persistence", aggfunc="mean")

    plt.figure(figsize=(7, 8))
    sns.heatmap(pivot, cmap="viridis", vmin=0, vmax=1)
    plt.title(title)
    plt.xlabel("Ligand")
    plt.ylabel("Residue")
    plt.tight_layout()
    plt.savefig(outpng, dpi=300)
    plt.close()

# ============================================================
# MAIN
# ============================================================

def main():
    os.makedirs("residence_results", exist_ok=True)

    systems = discover_systems()
    if not systems:
        raise FileNotFoundError("No liganded systems found.")

    all_persistence = []
    summary_rows = []

    for system in systems:
        print(f"Analyzing {system['protein']} / {system['ligand_name']}")

        res = analyze_contacts(system)

        time_ns = res["time_ns"]
        bound_series = res["bound_series"]
        persistence_df = res["persistence_df"]

        all_persistence.append(persistence_df)

        # dt between analyzed frames
        if len(time_ns) > 1:
            dt_ns = time_ns[1] - time_ns[0]
        else:
            dt_ns = 0.0

        tau_frames, S = survival_probability(bound_series)
        lag_ns = tau_frames * dt_ns if len(tau_frames) > 0 else np.array([0.0])

        tau_res = estimate_residence_time(tau_frames, S, dt_ns)
        diss_prob_final = 1.0 - S[-1] if len(S) > 0 else np.nan
        bound_fraction = np.mean(bound_series)

        # save raw bound series
        outbase = f"{system['protein']}_{system['ligand_key']}"
        pd.DataFrame({
            "time_ns": time_ns,
            "bound": bound_series
        }).to_csv(f"residence_results/{outbase}_bound_series.csv", index=False)

        persistence_df.to_csv(f"residence_results/{outbase}_contact_persistence.csv", index=False)

        plot_survival(
            lag_ns, S,
            f"residence_results/{outbase}_survival.png",
            f"{system['protein']} - {system['ligand_name']}"
        )
        plot_dissociation(
            lag_ns, S,
            f"residence_results/{outbase}_dissociation.png",
            f"{system['protein']} - {system['ligand_name']}"
        )

        summary_rows.append({
            "protein": system["protein"],
            "ligand": system["ligand_name"],
            "n_frames": res["n_frames"],
            "bound_fraction": bound_fraction,
            "residence_time_ns": tau_res,
            "final_dissociation_probability": diss_prob_final,
            "n_pocket_residues": len(res["pocket_residues"])
        })

    # combine persistence
    persistence_all = pd.concat(all_persistence, ignore_index=True)
    persistence_all.to_csv("residence_results/contact_persistence_all.csv", index=False)

    # summary
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv("residence_results/residence_summary.csv", index=False)

    print("\nResidence summary\n")
    print(summary_df.to_string(index=False))

    # heatmaps by protein
    for protein in summary_df["protein"].unique():
        sub = persistence_all[persistence_all["protein"] == protein]
        plot_persistence_heatmap(
            sub,
            f"residence_results/{protein}_contact_persistence_heatmap.png",
            f"{protein} Contact Persistence"
        )

if __name__ == "__main__":
    main()
