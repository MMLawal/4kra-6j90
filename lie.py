#!/usr/bin/env python3

# Run inside AmberTools25 environment

import os
import glob
import subprocess
import numpy as np
import pandas as pd

# =========================================================
# USER SETTINGS
# =========================================================

TRAJ_STRIDE = 1
VDW_CUTOFF = 12.0
ELEC_CUTOFF = 12.0

# residue selections
LIGAND_MAP = {
    "4kra": {"protein": "1-341", "ligand": "342"},
    "dnagyrase": {"protein": "1-390", "ligand": "391"},
}

# =========================================================
# DISCOVER SYSTEMS
# =========================================================

def discover_systems(base_dirs=("4kra","dnagyrase")):

    systems = []

    for protein in base_dirs:

        for lig_dir in sorted(glob.glob(os.path.join(protein,"*"))):

            ligand = os.path.basename(lig_dir)

            prmtop = os.path.join(lig_dir,"step3_input.parm7")
            dcd = os.path.join(lig_dir,"step1_production.dcd")

            if not os.path.exists(prmtop):
                continue

            if not os.path.exists(dcd):
                continue

            systems.append({
                "protein":protein,
                "ligand":ligand,
                "dir":lig_dir,
                "prmtop":prmtop,
                "traj":dcd
            })

    return systems


# =========================================================
# COMPUTE LIE
# =========================================================

def compute_interaction_energy(sys, workdir="lie_output"):

    os.makedirs(workdir, exist_ok=True)

    protein = sys["protein"]
    ligand = sys["ligand"]
    prmtop = sys["prmtop"]
    dcd = sys["traj"]

    # correct dictionary access
    res_map = LIGAND_MAP[protein]

    protein_sel = res_map["protein"]
    ligand_sel = res_map["ligand"]

    prefix = f"{protein}_{ligand}"

    lie_in = os.path.join(workdir, f"{prefix}_lie.in")
    lie_out = os.path.join(workdir, f"{prefix}_lie.dat")

    # build cpptraj script
    with open(lie_in, "w") as f:

        f.write(f"parm {prmtop}\n")
        f.write(f"trajin {dcd} 1 last {TRAJ_STRIDE}\n")

        f.write(
                f"lie :{ligand_sel} :{protein_sel} "
            f"out {lie_out} "
            f"cutvdw {VDW_CUTOFF} "
            f"cutelec {ELEC_CUTOFF}\n"
        )

        f.write("run\n")

    # run cpptraj
    try:

        subprocess.run(
            ["cpptraj", "-i", lie_in],
            check=True
        )

    except subprocess.CalledProcessError:

        print(f"cpptraj failed for {prefix}")
        return None

    # parse results
    if not os.path.exists(lie_out):
        print(f"Missing output {lie_out}")
        return None

    data = np.loadtxt(lie_out)

    vdw = data[:,1]
    ele = data[:,2]

    return {
        "protein":protein,
        "ligand":ligand,
        "vdw_mean":np.mean(vdw),
        "ele_mean":np.mean(ele),
        "LIE_total":np.mean(vdw + ele)
    }


# =========================================================
# RUN PIPELINE
# =========================================================

def run_pipeline():

    systems = discover_systems()

    print(f"\nDiscovered {len(systems)} systems\n")

    results = []

    for sys in systems:

        print(f"Running LIE for {sys['protein']} {sys['ligand']}")

        res = compute_interaction_energy(sys)

        if res:
            results.append(res)

    df = pd.DataFrame(results)

    df.to_csv("lie_energy_results.csv", index=False)

    print("\nLIE results\n")
    print(df)

    return df


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    run_pipeline()
