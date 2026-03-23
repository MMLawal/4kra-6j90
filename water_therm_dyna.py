#!/usr/bin/env python3

import os
import glob
import numpy as np
import pandas as pd
import MDAnalysis as mda
from MDAnalysis.lib.distances import distance_array
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from MDAnalysis.transformations import unwrap, center_in_box, wrap

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
TRAJECTORY_PATTERN = "step*_production.dcd"

# ============================================================
# PARAMETERS
# ============================================================

GRID_SPACING = 1.0       # Å
POCKET_CUTOFF = 5.0      # Å around ligand
GRID_PADDING = 4.0       # Å beyond pocket bounding box
KT = 0.593               # kcal/mol at 300 K
BULK_DENSITY = 0.0334    # waters / Å^3
STRIDE = 10              # analyze every 10th frame
WATER_SELECTION = "(resname WAT TIP3 HOH) and name O"

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

            prmtop = os.path.join(lig_dir, TOPOLOGY_FILE)
            trajs = sorted(glob.glob(os.path.join(lig_dir, TRAJECTORY_PATTERN)))

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
# PREPARE TRAJECTORY
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
# POCKET DEFINITION
# ============================================================

def define_pocket_atoms(u, ligand_resid):
    """
    Pocket = protein heavy atoms within POCKET_CUTOFF Å of ligand heavy atoms
    in the first frame.
    """
    ligand = u.select_atoms(f"resid {ligand_resid} and not name H*")
    protein = u.select_atoms("protein and not name H*")

    d = distance_array(protein.positions, ligand.positions)
    close = np.any(d <= POCKET_CUTOFF, axis=1)

    pocket_atoms = protein[close]

    if len(pocket_atoms) == 0:
        raise ValueError(f"No pocket atoms found within {POCKET_CUTOFF} Å of ligand resid {ligand_resid}")

    return pocket_atoms

# ============================================================
# GRID GENERATION
# ============================================================

def generate_grid(coords):
    min_xyz = coords.min(axis=0) - GRID_PADDING
    max_xyz = coords.max(axis=0) + GRID_PADDING

    xs = np.arange(min_xyz[0], max_xyz[0] + GRID_SPACING, GRID_SPACING)
    ys = np.arange(min_xyz[1], max_xyz[1] + GRID_SPACING, GRID_SPACING)
    zs = np.arange(min_xyz[2], max_xyz[2] + GRID_SPACING, GRID_SPACING)

    grid = np.zeros((len(xs), len(ys), len(zs)), dtype=float)

    return grid, xs, ys, zs

# ============================================================
# WATER DENSITY ACCUMULATION
# ============================================================

def accumulate_water_density(u, pocket_atoms):
    water = u.select_atoms(WATER_SELECTION)

    grid, xs, ys, zs = generate_grid(pocket_atoms.positions)

    n_frames = 0

    for ts in u.trajectory[::STRIDE]:
        pocket_coords = pocket_atoms.positions
        water_coords = water.positions

        d = distance_array(water_coords, pocket_coords)
        close = np.any(d <= POCKET_CUTOFF, axis=1)
        nearby_waters = water_coords[close]

        for w in nearby_waters:
            ix = int((w[0] - xs[0]) / GRID_SPACING)
            iy = int((w[1] - ys[0]) / GRID_SPACING)
            iz = int((w[2] - zs[0]) / GRID_SPACING)

            if (
                0 <= ix < grid.shape[0]
                and 0 <= iy < grid.shape[1]
                and 0 <= iz < grid.shape[2]
            ):
                grid[ix, iy, iz] += 1.0

        n_frames += 1

    if n_frames == 0:
        raise ValueError("No frames analyzed; check trajectory and stride.")

    grid /= n_frames

    return grid, xs, ys, zs, n_frames

# ============================================================
# DENSITY -> FREE ENERGY
# ============================================================

def density_to_free_energy(grid):
    rho = grid / (GRID_SPACING ** 3)

    with np.errstate(divide="ignore", invalid="ignore"):
        deltaG = -KT * np.log(rho / BULK_DENSITY)

    deltaG[np.isinf(deltaG)] = np.nan

    return deltaG, rho

# ============================================================
# WATER THERMODYNAMIC DESCRIPTORS
# ============================================================

def summarize_water_thermodynamics(deltaG, rho):
    valid = ~np.isnan(deltaG)

    if np.sum(valid) == 0:
        return {
            "mean_dG_water": np.nan,
            "median_dG_water": np.nan,
            "fraction_favorable": np.nan,
            "fraction_unfavorable": np.nan,
            "mean_favorable_dG": np.nan,
            "mean_unfavorable_dG": np.nan,
            "n_valid_voxels": 0
        }

    values = deltaG[valid]

    favorable = values < 0.0
    unfavorable = values > 0.0

    return {
        "mean_dG_water": float(np.nanmean(values)),
        "median_dG_water": float(np.nanmedian(values)),
        "fraction_favorable": float(np.sum(favorable) / len(values)),
        "fraction_unfavorable": float(np.sum(unfavorable) / len(values)),
        "mean_favorable_dG": float(np.nanmean(values[favorable])) if np.any(favorable) else np.nan,
        "mean_unfavorable_dG": float(np.nanmean(values[unfavorable])) if np.any(unfavorable) else np.nan,
        "n_valid_voxels": int(len(values))
    }

# ============================================================
# VISUALIZATION
# ============================================================

def plot_pocket_map(deltaG, protein, ligand_name):
    slice_z = deltaG.shape[2] // 2

    plt.figure(figsize=(6, 5))
    plt.imshow(
        gaussian_filter(deltaG[:, :, slice_z], sigma=1),
        cmap="coolwarm",
        origin="lower",
        aspect="auto"
    )
    plt.colorbar(label="ΔG_water (kcal/mol)")
    plt.title(f"{protein} - {ligand_name} water thermodynamic map")
    plt.xlabel("Grid X")
    plt.ylabel("Grid Y")
    plt.tight_layout()
    plt.savefig(f"GIST_map_{protein}_{ligand_name}.png", dpi=300)
    plt.close()

# ============================================================
# MAIN ANALYSIS
# ============================================================

def analyze_system(sys):
    print(f"Analyzing {sys['protein']} / {sys['ligand_name']}")

    u = mda.Universe(sys["prmtop"], sys["trajs"])
    u = prepare_trajectory(u)

    # pocket defined from first frame
    pocket_atoms = define_pocket_atoms(u, sys["ligand_resid"])

    grid, xs, ys, zs, n_frames = accumulate_water_density(u, pocket_atoms)
    deltaG, rho = density_to_free_energy(grid)

    plot_pocket_map(deltaG, sys["protein"], sys["ligand_name"])

    np.save(f"GIST_grid_{sys['protein']}_{sys['ligand_key']}.npy", deltaG)
    np.save(f"GIST_density_{sys['protein']}_{sys['ligand_key']}.npy", rho)

    summary = summarize_water_thermodynamics(deltaG, rho)
    summary.update({
        "protein": sys["protein"],
        "ligand": sys["ligand_name"],
        "ligand_key": sys["ligand_key"],
        "n_frames_analyzed": n_frames,
        "grid_shape": str(deltaG.shape)
    })

    return summary

# ============================================================
# RUN PIPELINE
# ============================================================

def main():
    systems = discover_systems()

    if not systems:
        raise FileNotFoundError("No valid systems found.")

    summaries = []

    for sys in systems:
        try:
            result = analyze_system(sys)
            summaries.append(result)
        except Exception as e:
            print(f"Failed for {sys['protein']} / {sys['ligand_name']}: {e}")

    df = pd.DataFrame(summaries)
    df.to_csv("GIST_water_thermodynamics_summary.csv", index=False)

    print("\nWater thermodynamics summary\n")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
