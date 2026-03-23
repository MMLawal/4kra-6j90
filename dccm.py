#!/usr/bin/env python3

"""
DCCM analysis for 4kra and dnagyrase systems

Computes residue-wise dynamical cross-correlation matrices (DCCM)
from C-alpha fluctuations and compares ligand-bound systems vs apo.

Outputs per system:
- dccm_<protein>_<ligand>.npy
- dccm_<protein>_<ligand>.csv
- dccm_<protein>_<ligand>.png

Comparison outputs per protein:
- dccm_difference_<protein>_<ligand>_minus_apo.npy
- dccm_difference_<protein>_<ligand>_minus_apo.png

Directory layout expected:
4kra/
   mangi/
   cipro/
   azad/
   protein/

dnagyrase/
   mangi/
   cipro/
   azad/
   protein/

Files expected in each system directory:
- step3_input.parm7
- step1_production.dcd  (or step*_production.dcd)
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import MDAnalysis as mda
from MDAnalysis.analysis import align
from MDAnalysis.transformations import unwrap, center_in_box, wrap

# ============================================================
# USER SETTINGS
# ============================================================

BASE_DIRS = ("4kra", "dnagyrase")

SYSTEMS = {
    "mangi": "Mangiferin",
    "cipro": "Ciprofloxacin",
    "azad": "Azadirachtin",
    "protein": "Apo",
}

TOPOLOGY_FILE = "step3_input.parm7"
TRAJ_PATTERN = "step*_production.dcd"

# stride for memory/performance
STRIDE = 10

# atom selection for DCCM
DCCM_SELECTION = "protein and name CA"

# alignment selection
ALIGN_SELECTION = "protein and name CA"

OUTDIR = "dccm_results"

# ============================================================
# DISCOVER SYSTEMS
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
            })

    return systems

# ============================================================
# TRAJECTORY PREPARATION
# ============================================================

def prepare_universe(prmtop, trajs):
    u = mda.Universe(prmtop, trajs)

    protein = u.select_atoms("protein")
    nonprotein = u.select_atoms("not protein")

    transforms = [
        unwrap(protein),
        center_in_box(protein, center="geometry", wrap=False),
        wrap(nonprotein, compound="residues"),
    ]
    u.trajectory.add_transformations(*transforms)

    return u

# ============================================================
# EXTRACT ALIGNED CA COORDINATES
# ============================================================

def extract_aligned_ca_coords(u, stride=10):
    ca = u.select_atoms(DCCM_SELECTION)

    if len(ca) == 0:
        raise ValueError("No C-alpha atoms found for DCCM selection.")

    # use first frame as reference
    ref = mda.Universe(u.filename, u.trajectory.filename if hasattr(u.trajectory, "filename") else None)
    # safer: build reference from current universe first frame
    u.trajectory[0]
    ref_coords = ca.positions.copy()

    coords = []
    resid = ca.resids.copy()
    resname = ca.resnames.copy()

    for ts in u.trajectory[::stride]:
        mobile = ca.positions.copy()

        # Kabsch alignment to first-frame reference
        mobile_center = mobile.mean(axis=0)
        ref_center = ref_coords.mean(axis=0)

        X = mobile - mobile_center
        Y = ref_coords - ref_center

        C = X.T @ Y
        V, S, Wt = np.linalg.svd(C)
        d = np.sign(np.linalg.det(V @ Wt))
        D = np.diag([1.0, 1.0, d])
        R = V @ D @ Wt

        aligned = X @ R + ref_center
        coords.append(aligned)

    coords = np.array(coords)  # (n_frames, n_res, 3)
    return coords, resid, resname

# ============================================================
# DCCM CALCULATION
# ============================================================

def compute_dccm(coords):
    """
    coords shape: (n_frames, n_residues, 3)

    DCCM_ij = <Δri · Δrj> / sqrt(<|Δri|^2><|Δrj|^2>)
    """
    mean_coords = coords.mean(axis=0)
    fluctuations = coords - mean_coords  # (T, N, 3)

    # flatten dot products over time
    n_frames, n_res, _ = fluctuations.shape
    dccm = np.zeros((n_res, n_res), dtype=np.float64)

    # variance terms
    var = np.mean(np.sum(fluctuations * fluctuations, axis=2), axis=0)  # (N,)

    for i in range(n_res):
        fi = fluctuations[:, i, :]  # (T, 3)
        for j in range(i, n_res):
            fj = fluctuations[:, j, :]
            num = np.mean(np.sum(fi * fj, axis=1))
            den = np.sqrt(var[i] * var[j])

            val = 0.0 if den == 0 else num / den
            dccm[i, j] = val
            dccm[j, i] = val

    return dccm

# ============================================================
# SAVE MATRIX
# ============================================================

def save_dccm_outputs(dccm, resid, resname, protein, ligand_key, ligand_name):
    os.makedirs(OUTDIR, exist_ok=True)

    tag = f"{protein}_{ligand_key}"

    np.save(os.path.join(OUTDIR, f"dccm_{tag}.npy"), dccm)

    labels = [f"{rn}{ri}" for rn, ri in zip(resname, resid)]
    df = pd.DataFrame(dccm, index=labels, columns=labels)
    df.to_csv(os.path.join(OUTDIR, f"dccm_{tag}.csv"))

    plt.figure(figsize=(8, 7))
    im = plt.imshow(dccm, cmap="coolwarm", vmin=-1, vmax=1, origin="lower", aspect="auto")
    plt.colorbar(im, label="Cross-correlation")
    plt.title(f"DCCM: {protein} - {ligand_name}")
    plt.xlabel("Residue index")
    plt.ylabel("Residue index")

    tick_positions = np.linspace(0, len(resid) - 1, min(10, len(resid)), dtype=int)
    tick_labels = [str(resid[i]) for i in tick_positions]
    plt.xticks(tick_positions, tick_labels, rotation=45)
    plt.yticks(tick_positions, tick_labels)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, f"dccm_{tag}.png"), dpi=300)
    plt.close()

# ============================================================
# DIFFERENCE MAPS VS APO
# ============================================================

def save_difference_map(dccm_lig, dccm_apo, resid, protein, ligand_key, ligand_name):
    delta = dccm_lig - dccm_apo
    tag = f"{protein}_{ligand_key}_minus_apo"

    np.save(os.path.join(OUTDIR, f"dccm_difference_{tag}.npy"), delta)

    plt.figure(figsize=(8, 7))
    im = plt.imshow(delta, cmap="bwr", vmin=-1, vmax=1, origin="lower", aspect="auto")
    plt.colorbar(im, label="Δ cross-correlation")
    plt.title(f"DCCM Difference: {protein} {ligand_name} - Apo")
    plt.xlabel("Residue index")
    plt.ylabel("Residue index")

    tick_positions = np.linspace(0, len(resid) - 1, min(10, len(resid)), dtype=int)
    tick_labels = [str(resid[i]) for i in tick_positions]
    plt.xticks(tick_positions, tick_labels, rotation=45)
    plt.yticks(tick_positions, tick_labels)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, f"dccm_difference_{tag}.png"), dpi=300)
    plt.close()

# ============================================================
# SUMMARY METRICS
# ============================================================

def summarize_dccm(dccm, protein, ligand_name):
    upper = dccm[np.triu_indices_from(dccm, k=1)]
    return {
        "protein": protein,
        "ligand": ligand_name,
        "mean_corr": float(np.mean(upper)),
        "mean_abs_corr": float(np.mean(np.abs(upper))),
        "fraction_positive_corr": float(np.mean(upper > 0)),
        "fraction_negative_corr": float(np.mean(upper < 0)),
        "strong_positive_corr_fraction": float(np.mean(upper > 0.5)),
        "strong_negative_corr_fraction": float(np.mean(upper < -0.5)),
    }

# ============================================================
# MAIN
# ============================================================

def main():
    os.makedirs(OUTDIR, exist_ok=True)

    systems = discover_systems()
    if not systems:
        raise FileNotFoundError("No valid systems found.")

    dccm_store = {}
    resid_store = {}
    summaries = []

    for sys in systems:
        protein = sys["protein"]
        lig_key = sys["ligand_key"]
        lig_name = sys["ligand_name"]

        print(f"Analyzing DCCM for {protein} / {lig_name}")

        try:
            u = prepare_universe(sys["prmtop"], sys["trajs"])
            coords, resid, resname = extract_aligned_ca_coords(u, stride=STRIDE)
            dccm = compute_dccm(coords)

            dccm_store[(protein, lig_key)] = dccm
            resid_store[protein] = (resid, resname)

            save_dccm_outputs(dccm, resid, resname, protein, lig_key, lig_name)
            summaries.append(summarize_dccm(dccm, protein, lig_name))

        except Exception as e:
            print(f"Failed for {protein} / {lig_name}: {e}")

    # save summary
    if summaries:
        summary_df = pd.DataFrame(summaries)
        summary_df.to_csv(os.path.join(OUTDIR, "dccm_summary.csv"), index=False)

    # ligand vs apo difference maps
    for protein in BASE_DIRS:
        if (protein, "protein") not in dccm_store:
            print(f"No apo DCCM found for {protein}; skipping difference maps.")
            continue

        dccm_apo = dccm_store[(protein, "protein")]
        resid, resname = resid_store[protein]

        for lig_key in ("mangi", "cipro", "azad"):
            if (protein, lig_key) not in dccm_store:
                continue

            dccm_lig = dccm_store[(protein, lig_key)]
            lig_name = SYSTEMS[lig_key]
            save_difference_map(dccm_lig, dccm_apo, resid, protein, lig_key, lig_name)

    print(f"\nDone. Outputs written to: {OUTDIR}")

if __name__ == "__main__":
    main()
