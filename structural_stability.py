#!/usr/bin/env python3

import os
import glob
import json
import math
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import MDAnalysis as mda
from MDAnalysis.lib.distances import distance_array
from MDAnalysis.transformations import unwrap, center_in_box, wrap
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore")

# ============================================================
# USER SETTINGS
# ============================================================

BASE_DIRS = ("4kra", "dnagyrase")

SYSTEMS = {
    "mangi": "Mangiferin",
    "cipro": "Ciprofloxacin",
    "azad": "Azadirachtin",
}

LIGAND_MAP = {
    "4kra": {"protein": "1-341", "ligand": 342},
    "dnagyrase": {"protein": "1-390", "ligand": 391},
}

TOPOLOGY_FILE = "step3_input.parm7"
TRAJ_PATTERN = "step*_production.dcd"

# analysis controls
STRIDE = 10
POCKET_CUTOFF = 5.0         # pocket residues = within 5 Å of ligand in first frame
PROBE_RADIUS = 1.4          # SASA probe radius, Å
N_SPHERE = 96               # SASA sphere points
OUTDIR = "structural_stability_results"

# Thermodynamic constants
kB = 1.380649e-23           # J/K
NA = 6.02214076e23          # 1/mol
HBAR = 1.054571817e-34      # J*s
T = 300.0                   # K
ANGSTROM_TO_M = 1e-10
AMU_TO_KG = 1.66053906660e-27

# minimal vdW radii table for SASA (Å)
VDW_RADII = {
    "H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "F": 1.47,
    "P": 1.80, "S": 1.80, "CL": 1.75, "BR": 1.85, "I": 1.98,
    "NA": 2.27, "MG": 1.73, "ZN": 1.39, "CA": 2.31, "FE": 1.56
}

# ============================================================
# DISCOVERY
# ============================================================

def discover_systems():
    systems = []
    for protein in BASE_DIRS:
        for lig_dir in sorted(glob.glob(os.path.join(protein, "*"))):
            lig_key = os.path.basename(lig_dir)
            if lig_key not in SYSTEMS:
                continue

            prmtop = os.path.join(lig_dir, TOPOLOGY_FILE)
            trajs = sorted(glob.glob(os.path.join(lig_dir, TRAJ_PATTERN)))
            if not os.path.exists(prmtop) or not trajs:
                continue

            systems.append({
                "protein": protein,
                "ligand_key": lig_key,
                "ligand_name": SYSTEMS[lig_key],
                "dir": lig_dir,
                "prmtop": prmtop,
                "trajs": trajs,
                "ligand_resid": LIGAND_MAP[protein]["ligand"],
            })
    return systems

# ============================================================
# TRAJECTORY PREP
# ============================================================

def prepare_trajectory(u):
    protein = u.select_atoms("protein")
    solvent_ions = u.select_atoms("not protein")

    transforms = [
        unwrap(protein),
        center_in_box(protein, center="geometry", wrap=False),
        wrap(solvent_ions, compound="residues")
    ]
    u.trajectory.add_transformations(*transforms)
    return u

# ============================================================
# GEOMETRY HELPERS
# ============================================================

def kabsch_align(mobile, ref):
    mobile_cent = mobile.mean(axis=0)
    ref_cent = ref.mean(axis=0)

    X = mobile - mobile_cent
    Y = ref - ref_cent

    C = X.T @ Y
    V, S, Wt = np.linalg.svd(C)
    d = np.sign(np.linalg.det(V @ Wt))
    D = np.diag([1.0, 1.0, d])
    R = V @ D @ Wt

    aligned = X @ R + ref_cent
    return aligned

def rmsd(coords_a, coords_b):
    diff = coords_a - coords_b
    return np.sqrt(np.mean(np.sum(diff * diff, axis=1)))

# ============================================================
# POCKET DEFINITION
# ============================================================

def define_pocket_residues(u, ligand_resid, cutoff=5.0):
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
        raise ValueError("No pocket residues found.")
    return pocket

# ============================================================
# ELEMENT / RADII / MASSES
# ============================================================

def guess_element(atom):
    if atom.element and str(atom.element).strip():
        return str(atom.element).strip().upper()

    name = atom.name.strip().upper()
    if len(name) >= 2 and name[:2] in VDW_RADII:
        return name[:2]
    return name[0]

def atom_radius(atom):
    el = guess_element(atom)
    return VDW_RADII.get(el, 1.70)

def atom_mass_kg(atom):
    mass = atom.mass if atom.mass is not None and atom.mass > 0 else 12.011
    return mass * AMU_TO_KG

# ============================================================
# SPHERE POINTS FOR SASA
# ============================================================

def fibonacci_sphere(n=96):
    pts = []
    phi = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(n):
        y = 1 - (i / float(n - 1)) * 2
        radius = math.sqrt(max(0.0, 1 - y * y))
        theta = phi * i
        x = math.cos(theta) * radius
        z = math.sin(theta) * radius
        pts.append((x, y, z))
    return np.array(pts, dtype=float)

SPHERE_POINTS = fibonacci_sphere(N_SPHERE)

# ============================================================
# SASA
# ============================================================

def shrake_rupley_subset(all_coords, all_radii, subset_idx, probe_radius=1.4, n_sphere=96):
    """
    Compute SASA for selected atoms, considering occlusion by all atoms.
    Returns total SASA in Å^2 for the subset.
    """
    if len(subset_idx) == 0:
        return 0.0

    sphere_pts = SPHERE_POINTS if n_sphere == N_SPHERE else fibonacci_sphere(n_sphere)
    expanded = all_radii + probe_radius
    tree = cKDTree(all_coords)

    total_sasa = 0.0

    for i in subset_idx:
        r_i = expanded[i]
        center = all_coords[i]
        test_points = center + sphere_pts * r_i

        neighbor_idx = tree.query_ball_point(center, r=r_i + np.max(expanded))
        neighbor_idx = [j for j in neighbor_idx if j != i]

        if len(neighbor_idx) == 0:
            exposed_fraction = 1.0
        else:
            neighbors = all_coords[neighbor_idx]
            neigh_r = expanded[neighbor_idx]

            exposed = np.ones(len(test_points), dtype=bool)
            for j, p in enumerate(test_points):
                d2 = np.sum((neighbors - p) ** 2, axis=1)
                if np.any(d2 < (neigh_r ** 2)):
                    exposed[j] = False

            exposed_fraction = np.sum(exposed) / len(exposed)

        area = 4.0 * math.pi * (r_i ** 2) * exposed_fraction
        total_sasa += area

    return total_sasa

# ============================================================
# ENTROPY CALCULATIONS
# ============================================================

def covariance_from_coords(coords):
    """
    coords: (n_frames, n_atoms, 3)
    Returns Cartesian covariance matrix shape (3N, 3N)
    """
    X = coords.reshape(coords.shape[0], -1)
    X = X - X.mean(axis=0, keepdims=True)
    C = np.cov(X, rowvar=False, bias=False)
    return C

def mass_weighted_covariance(coords, masses_kg):
    """
    coords in Å -> convert to m
    masses_kg shape (N,)
    Returns mass-weighted covariance in SI units.
    """
    X = coords * ANGSTROM_TO_M
    X = X.reshape(X.shape[0], -1)
    X = X - X.mean(axis=0, keepdims=True)

    mw = np.repeat(np.sqrt(masses_kg), 3)
    Xmw = X * mw[None, :]
    Cmw = np.cov(Xmw, rowvar=False, bias=False)
    return Cmw

def quasiharmonic_entropy(coords, masses_kg, temperature=300.0):
    """
    Quasiharmonic entropy from mass-weighted covariance eigenvalues.
    Returns J/mol/K.
    """
    Cmw = mass_weighted_covariance(coords, masses_kg)

    eigvals = np.linalg.eigvalsh(Cmw)
    eigvals = eigvals[eigvals > 1e-30]

    if len(eigvals) == 0:
        return np.nan

    # omega_i = sqrt(k_B T / lambda_i)
    omega = np.sqrt((kB * temperature) / eigvals)

    alpha = (HBAR * omega) / (kB * temperature)

    # quantum harmonic entropy
    S_modes = kB * (alpha / (np.exp(alpha) - 1.0) - np.log(1.0 - np.exp(-alpha)))
    S = np.sum(S_modes) * NA
    return S

def schlitter_entropy(coords, masses_kg, temperature=300.0):
    """
    Schlitter entropy lower bound from mass-weighted covariance.
    Returns J/mol/K.
    """
    Cmw = mass_weighted_covariance(coords, masses_kg)

    factor = (kB * temperature * math.e ** 2) / (HBAR ** 2)
    M = np.eye(Cmw.shape[0]) + factor * Cmw

    sign, logdet = np.linalg.slogdet(M)
    if sign <= 0:
        return np.nan

    S = 0.5 * kB * logdet * NA
    return S

# ============================================================
# MAIN ANALYSIS PER SYSTEM
# ============================================================

def analyze_system(sysinfo):
    print(f"Analyzing {sysinfo['protein']} / {sysinfo['ligand_name']}")

    outdir = os.path.join(OUTDIR, f"{sysinfo['protein']}_{sysinfo['ligand_key']}")
    os.makedirs(outdir, exist_ok=True)

    u = mda.Universe(sysinfo["prmtop"], sysinfo["trajs"])
    u = prepare_trajectory(u)

    ligand = u.select_atoms(f"resid {sysinfo['ligand_resid']} and not name H*")
    if len(ligand) == 0:
        raise ValueError("No ligand atoms found.")

    pocket_info = define_pocket_residues(u, sysinfo["ligand_resid"], cutoff=POCKET_CUTOFF)
    pocket_resids = [r for r, _ in pocket_info]
    pocket = u.select_atoms("protein and not name H* and resid " + " ".join(map(str, pocket_resids)))

    protein_ca = u.select_atoms("protein and name CA")

    # reference coordinates
    u.trajectory[0]
    ref_protein_ca = protein_ca.positions.copy()
    ref_ligand = ligand.positions.copy()
    ref_pocket = pocket.positions.copy()

    # atom radii / masses for SASA and entropy
    union_atoms = u.select_atoms("protein and not name H* or resid %d and not name H*" % sysinfo["ligand_resid"])
    union_radii = np.array([atom_radius(a) for a in union_atoms], dtype=float)

    ligand_indices_in_union = []
    pocket_indices_in_union = []
    ligand_set = set(ligand.indices.tolist())
    pocket_set = set(pocket.indices.tolist())

    for idx, atom in enumerate(union_atoms):
        if atom.index in ligand_set:
            ligand_indices_in_union.append(idx)
        if atom.index in pocket_set:
            pocket_indices_in_union.append(idx)

    ligand_masses = np.array([atom_mass_kg(a) for a in ligand], dtype=float)

    # storage
    time_ns = []
    ligand_rmsd_vals = []
    pocket_rmsd_vals = []
    ligand_sasa_vals = []
    pocket_sasa_vals = []
    ligand_aligned_coords = []

    for iframe, ts in enumerate(u.trajectory[::STRIDE]):
        # align whole protein to first frame using CA atoms
        current_ca = protein_ca.positions.copy()
        aligned_ca = kabsch_align(current_ca, ref_protein_ca)

        # derive rigid transform from protein CA alignment using centroids + Kabsch again
        mob_cent = current_ca.mean(axis=0)
        ref_cent = ref_protein_ca.mean(axis=0)
        X = current_ca - mob_cent
        Y = ref_protein_ca - ref_cent
        C = X.T @ Y
        V, S, Wt = np.linalg.svd(C)
        d = np.sign(np.linalg.det(V @ Wt))
        D = np.diag([1.0, 1.0, d])
        R = V @ D @ Wt

        # apply same transform to pocket and ligand
        lig_coords = (ligand.positions.copy() - mob_cent) @ R + ref_cent
        pocket_coords = (pocket.positions.copy() - mob_cent) @ R + ref_cent

        # ligand internal RMSD:
        # align ligand to its own first-frame ligand reference to remove translation/rotation
        lig_internal = kabsch_align(ligand.positions.copy(), ref_ligand)
        lig_rmsd = rmsd(lig_internal, ref_ligand)

        # pocket RMSD after protein alignment
        pocket_r = rmsd(pocket_coords, ref_pocket)

        # SASA in complex environment
        union_coords = union_atoms.positions.copy()
        ligand_sasa = shrake_rupley_subset(
            union_coords, union_radii, ligand_indices_in_union,
            probe_radius=PROBE_RADIUS, n_sphere=N_SPHERE
        )
        pocket_sasa = shrake_rupley_subset(
            union_coords, union_radii, pocket_indices_in_union,
            probe_radius=PROBE_RADIUS, n_sphere=N_SPHERE
        )

        # save
        time_ns.append(ts.time / 1000.0 if ts.time is not None else iframe)
        ligand_rmsd_vals.append(lig_rmsd)
        pocket_rmsd_vals.append(pocket_r)
        ligand_sasa_vals.append(ligand_sasa)
        pocket_sasa_vals.append(pocket_sasa)
        ligand_aligned_coords.append(lig_internal)

    ligand_aligned_coords = np.array(ligand_aligned_coords)

    # entropy
    qh_S = quasiharmonic_entropy(ligand_aligned_coords, ligand_masses, temperature=T)
    schlitter_S = schlitter_entropy(ligand_aligned_coords, ligand_masses, temperature=T)

    # per-frame dataframe
    frame_df = pd.DataFrame({
        "time_ns": time_ns,
        "ligand_internal_rmsd_A": ligand_rmsd_vals,
        "pocket_rmsd_A": pocket_rmsd_vals,
        "ligand_sasa_A2": ligand_sasa_vals,
        "pocket_sasa_A2": pocket_sasa_vals
    })
    frame_df.to_csv(os.path.join(outdir, "structural_metrics_timeseries.csv"), index=False)

    # summary
    summary = {
        "protein": sysinfo["protein"],
        "ligand": sysinfo["ligand_name"],
        "n_frames": len(frame_df),
        "n_pocket_residues": len(pocket_resids),
        "ligand_internal_rmsd_mean_A": float(np.mean(ligand_rmsd_vals)),
        "ligand_internal_rmsd_std_A": float(np.std(ligand_rmsd_vals, ddof=1)),
        "pocket_rmsd_mean_A": float(np.mean(pocket_rmsd_vals)),
        "pocket_rmsd_std_A": float(np.std(pocket_rmsd_vals, ddof=1)),
        "ligand_sasa_mean_A2": float(np.mean(ligand_sasa_vals)),
        "ligand_sasa_std_A2": float(np.std(ligand_sasa_vals, ddof=1)),
        "pocket_sasa_mean_A2": float(np.mean(pocket_sasa_vals)),
        "pocket_sasa_std_A2": float(np.std(pocket_sasa_vals, ddof=1)),
        "quasiharmonic_entropy_J_mol_K": float(qh_S) if not np.isnan(qh_S) else np.nan,
        "schlitter_entropy_J_mol_K": float(schlitter_S) if not np.isnan(schlitter_S) else np.nan,
        "pocket_residues": ",".join([f"{rn}{ri}" for ri, rn in pocket_info])
    }

    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # plots
    # 1. RMSD traces
    plt.figure(figsize=(7, 5))
    plt.plot(time_ns, ligand_rmsd_vals, label="Ligand internal RMSD")
    plt.plot(time_ns, pocket_rmsd_vals, label="Pocket RMSD")
    plt.xlabel("Time (ns)")
    plt.ylabel("RMSD (Å)")
    plt.title(f"{sysinfo['protein']} - {sysinfo['ligand_name']}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "rmsd_ligand_vs_pocket.png"), dpi=300)
    plt.close()

    # 2. SASA traces
    plt.figure(figsize=(7, 5))
    plt.plot(time_ns, ligand_sasa_vals, label="Ligand SASA")
    plt.plot(time_ns, pocket_sasa_vals, label="Pocket SASA")
    plt.xlabel("Time (ns)")
    plt.ylabel("SASA (Å$^2$)")
    plt.title(f"{sysinfo['protein']} - {sysinfo['ligand_name']}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "sasa_ligand_vs_pocket.png"), dpi=300)
    plt.close()

    return summary

# ============================================================
# MAIN
# ============================================================

def main():
    os.makedirs(OUTDIR, exist_ok=True)
    systems = discover_systems()
    if not systems:
        raise FileNotFoundError("No systems found.")

    summaries = []
    for sysinfo in systems:
        try:
            res = analyze_system(sysinfo)
            summaries.append(res)
        except Exception as e:
            print(f"Failed for {sysinfo['protein']} / {sysinfo['ligand_name']}: {e}")

    if summaries:
        df = pd.DataFrame(summaries)
        df.to_csv(os.path.join(OUTDIR, "structural_stability_summary.csv"), index=False)

        print("\nStructural stability summary\n")
        print(df[[
            "protein", "ligand",
            "ligand_internal_rmsd_mean_A",
            "pocket_rmsd_mean_A",
            "ligand_sasa_mean_A2",
            "pocket_sasa_mean_A2",
            "quasiharmonic_entropy_J_mol_K",
            "schlitter_entropy_J_mol_K"
        ]].to_string(index=False))

if __name__ == "__main__":
    main()
