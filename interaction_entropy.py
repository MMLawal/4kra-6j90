#!/usr/bin/env python3

import os
import glob
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count

# ============================================================
# CONSTANTS
# ============================================================

BETA = 0.92
WINDOW = 50
NPROC = max(cpu_count() - 2, 1)

# ============================================================
# DISCOVER SYSTEMS
# ============================================================

def discover_systems(base_dirs=("4kra", "dnagyrase")):

    systems = []

    for protein in base_dirs:
        for lig_dir in sorted(glob.glob(os.path.join(protein, "*"))):
            ligand = os.path.basename(lig_dir)

            prmtop = os.path.join(lig_dir, "step3_input.parm7")
            traj = os.path.join(lig_dir, "step1_production.dcd")

            if not os.path.exists(prmtop):
                continue
            if not os.path.exists(traj):
                continue

            systems.append({
                "protein": protein,
                "ligand": ligand,
                "dir": lig_dir,
                "prmtop": prmtop,
                "traj": traj
            })

    return systems

# ============================================================
# LOAD LIE ENERGY FILE
# ============================================================

def load_lie_energy(file):
    data = np.loadtxt(file)
    vdw = data[:, 1]
    ele = data[:, 2]
    return vdw + ele

# ============================================================
# INTERACTION ENTROPY
# ============================================================

def interaction_entropy_with_sem(energy_series):

    IE_values = []

    for i in range(WINDOW, len(energy_series)):
        window = energy_series[i - WINDOW:i]
        dE = window - np.mean(window)
        val = np.log(np.mean(np.exp(BETA * dE)))
        IE_values.append(val)

    IE_values = np.array(IE_values, dtype=float)

    ie_mean = np.mean(IE_values)
    ie_sem = np.std(IE_values, ddof=1) / np.sqrt(len(IE_values)) if len(IE_values) > 1 else np.nan

    return ie_mean, ie_sem, IE_values

# ============================================================
# PROCESS SINGLE SYSTEM
# ============================================================

def process_system(system):

    protein = system["protein"]
    ligand = system["ligand"]

    lie_file = f"lie_output/{protein}_{ligand}_lie.dat"

    if not os.path.exists(lie_file):
        print(f"Missing {lie_file}")
        return None

    energies = load_lie_energy(lie_file)

    dE_MM_mean = np.mean(energies)
    dE_MM_sem = np.std(energies, ddof=1) / np.sqrt(len(energies)) if len(energies) > 1 else np.nan

    ie_mean, ie_sem, _ = interaction_entropy_with_sem(energies)

    return {
        "protein": protein,
        "ligand": ligand,
        "n_frames": int(len(energies)),
        "dE_MM_kcal_mol": dE_MM_mean,
        "dE_MM_sem": dE_MM_sem,
        "minus_TdS_kcal_mol": ie_mean,
        "minus_TdS_sem": ie_sem
    }

# ============================================================
# PARALLEL EXECUTION
# ============================================================

def run_pipeline():

    systems = discover_systems()

    print(f"\nDiscovered {len(systems)} systems\n")

    with Pool(NPROC) as pool:
        results = pool.map(process_system, systems)

    results = [r for r in results if r is not None]

    df = pd.DataFrame(results)
    df.to_csv("entropy_results_with_sem.csv", index=False)

    print("✓ Interaction entropy analysis complete")
    return df

# ============================================================
# MAIN
# ============================================================

def main():
    energy_df = run_pipeline()
    print("\n✓ Analysis Pipeline Complete\n")
    print(energy_df.to_string(index=False))

if __name__ == "__main__":
    main()
