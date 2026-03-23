#!/usr/bin/env python3

import os
import glob
import math
import json
import numpy as np
import pandas as pd
import MDAnalysis as mda
from MDAnalysis.lib.distances import distance_array
from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import HydrogenBondAnalysis
from MDAnalysis.transformations import unwrap, center_in_box, wrap

# Optional: networkx for ligand ring detection
try:
    import networkx as nx
    HAS_NX = True
except Exception:
    HAS_NX = False

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

STRIDE = 10
POCKET_CUTOFF = 5.0

# H-bond cutoffs
HBOND_DA_CUTOFF = 3.5
HBOND_ANGLE_CUTOFF = 150.0

# Salt bridge cutoff
SALTBRIDGE_CUTOFF = 4.0

# Hydrophobic contact cutoff
HYDROPHOBIC_CUTOFF = 4.5

# Pi-stacking cutoffs
PI_PI_CENTROID_CUTOFF = 5.5
PI_PI_PLANE_ANGLE_CUTOFF = 30.0      # parallel / face-to-face
PI_T_ANGLE_TARGET = 90.0              # T-shaped
PI_T_ANGLE_TOL = 30.0

OUTDIR = "interaction_fingerprint_results"

# ============================================================
# CHEMICAL DEFINITIONS
# ============================================================

# protein aromatic rings
PROTEIN_AROMATIC_RINGS = {
    "PHE": [["CG", "CD1", "CD2", "CE1", "CE2", "CZ"]],
    "TYR": [["CG", "CD1", "CD2", "CE1", "CE2", "CZ"]],
    "TRP": [
        ["CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2"],  # six-membered
        ["CG", "CD1", "NE1", "CE2", "CD2"],          # five-membered
    ],
    "HIS": [["CG", "ND1", "CD2", "CE1", "NE2"]],
    "HIE": [["CG", "ND1", "CD2", "CE1", "NE2"]],
    "HID": [["CG", "ND1", "CD2", "CE1", "NE2"]],
    "HIP": [["CG", "ND1", "CD2", "CE1", "NE2"]],
}

BASIC_RESNAMES = {"ARG", "LYS", "HIP", "HIS", "HIE", "HID"}
ACIDIC_RESNAMES = {"ASP", "GLU"}

# acidic/basic atoms for salt bridges
ACIDIC_ATOMS = {
    "ASP": ["OD1", "OD2"],
    "GLU": ["OE1", "OE2"],
}
BASIC_ATOMS = {
    "ARG": ["NH1", "NH2", "NE"],
    "LYS": ["NZ"],
    "HIP": ["ND1", "NE2"],
    "HIS": ["ND1", "NE2"],
    "HIE": ["ND1", "NE2"],
    "HID": ["ND1", "NE2"],
}

HYDROPHOBIC_ELEMENTS = {"C", "S"}

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
        wrap(solvent_ions, compound="residues"),
    ]
    u.trajectory.add_transformations(*transforms)
    return u

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
# HELPERS
# ============================================================

def atom_element(atom):
    if atom.element and str(atom.element).strip():
        return str(atom.element).strip().upper()
    name = atom.name.strip().upper()
    if len(name) >= 2 and name[:2] in {"CL", "BR", "NA", "MG", "ZN", "FE"}:
        return name[:2]
    return name[0]

def safe_unit(v):
    n = np.linalg.norm(v)
    if n < 1e-8:
        return None
    return v / n

def ring_centroid_and_normal(coords):
    centroid = coords.mean(axis=0)
    x = coords - centroid
    if len(coords) < 3:
        return centroid, None
    # plane normal via SVD
    _, _, vh = np.linalg.svd(x, full_matrices=False)
    normal = vh[-1]
    normal = safe_unit(normal)
    return centroid, normal

def angle_deg(u, v):
    if u is None or v is None:
        return None
    c = np.clip(np.abs(np.dot(u, v)), -1.0, 1.0)
    return np.degrees(np.arccos(c))

# ============================================================
# LIGAND RING DETECTION
# ============================================================

def detect_ligand_rings(ligand):
    """
    Detect 5/6-member cycles from bond graph.
    Requires bond info in topology and networkx.
    Returns list of atom index lists relative to ligand.atoms.
    """
    rings = []

    if not HAS_NX:
        return rings

    # map global atom index -> local index
    local = {a.index: i for i, a in enumerate(ligand.atoms)}

    G = nx.Graph()
    for i in range(len(ligand.atoms)):
        G.add_node(i)

    for bond in ligand.bonds:
        a = bond.atoms[0].index
        b = bond.atoms[1].index
        if a in local and b in local:
            ia = local[a]
            ib = local[b]
            G.add_edge(ia, ib)

    cycles = nx.cycle_basis(G)

    for cyc in cycles:
        if len(cyc) not in (5, 6):
            continue

        atoms = ligand.atoms[cyc]
        elems = [atom_element(a) for a in atoms]

        # aromatic-like heuristic: mostly C/N and planar-ish candidates
        if all(e in {"C", "N"} for e in elems):
            rings.append(cyc)

    return rings

# ============================================================
# PROTEIN RING DEFINITIONS
# ============================================================

def get_protein_ring_atoms(residue):
    rings = []
    resname = residue.resname.upper()

    if resname not in PROTEIN_AROMATIC_RINGS:
        return rings

    atom_lookup = {a.name: a for a in residue.atoms}
    for ring_names in PROTEIN_AROMATIC_RINGS[resname]:
        if all(name in atom_lookup for name in ring_names):
            rings.append([atom_lookup[name] for name in ring_names])

    return rings

# ============================================================
# H-BOND PERSISTENCE
# ============================================================

def compute_hbond_persistence(u, ligand_resid, pocket_resids, stride=10):
    """
    Uses MDAnalysis H-bond analysis, then filters to ligand<->pocket.
    """
    pocket_sel = "protein and resid " + " ".join(map(str, pocket_resids))
    ligand_sel = f"resid {ligand_resid}"

    h = HydrogenBondAnalysis(
        universe=u,
        donors_sel=None,
        hydrogens_sel=None,
        acceptors_sel=None,
        between=[(ligand_sel, pocket_sel), (pocket_sel, ligand_sel)],
        d_a_cutoff=HBOND_DA_CUTOFF,
        d_h_a_angle_cutoff=HBOND_ANGLE_CUTOFF,
        update_selections=True
    )
    h.run(step=stride)

    hbonds = h.results.hbonds
    n_frames = len(u.trajectory[::stride])

    # columns: frame, donor_ix, hydrogen_ix, acceptor_ix, distance, angle
    pocket_counts = {}

    for row in hbonds:
        donor_idx = int(row[1])
        acceptor_idx = int(row[3])

        donor_atom = u.atoms[donor_idx]
        acceptor_atom = u.atoms[acceptor_idx]

        # pocket residue is whichever atom belongs to protein
        pocket_atom = donor_atom if donor_atom.residue.resid in pocket_resids else acceptor_atom
        resid = int(pocket_atom.residue.resid)
        resname = pocket_atom.residue.resname

        pocket_counts[(resid, resname)] = pocket_counts.get((resid, resname), 0) + 1

    rows = []
    for (resid, resname), count in pocket_counts.items():
        rows.append({
            "resid": resid,
            "resname": resname,
            "interaction_type": "hbond",
            "persistence": count / n_frames
        })

    return pd.DataFrame(rows)

# ============================================================
# SALT BRIDGES
# ============================================================

def compute_saltbridge_persistence(u, ligand_resid, pocket_resids, stride=10):
    ligand = u.select_atoms(f"resid {ligand_resid}")
    pocket_residue_objs = [u.select_atoms(f"protein and resid {r}").residues[0] for r in pocket_resids]

    # classify ligand charged atoms heuristically
    ligand_basic = []
    ligand_acidic = []

    for atom in ligand.atoms:
        nm = atom.name.upper()
        el = atom_element(atom)

        # heuristic ligand charges from atom names/elements
        if el == "N":
            ligand_basic.append(atom)
        if el == "O":
            ligand_acidic.append(atom)

    counts = {}
    n_frames = 0

    for ts in u.trajectory[::stride]:
        n_frames += 1

        for res in pocket_residue_objs:
            resname = res.resname.upper()
            resid = int(res.resid)

            present = False

            if resname in ACIDIC_ATOMS and ligand_basic:
                acidic_atoms = [a for a in res.atoms if a.name in ACIDIC_ATOMS[resname]]
                if acidic_atoms:
                    d = distance_array(
                        np.array([a.position for a in acidic_atoms]),
                        np.array([a.position for a in ligand_basic])
                    )
                    if np.min(d) <= SALTBRIDGE_CUTOFF:
                        present = True

            elif resname in BASIC_ATOMS and ligand_acidic:
                basic_atoms = [a for a in res.atoms if a.name in BASIC_ATOMS[resname]]
                if basic_atoms:
                    d = distance_array(
                        np.array([a.position for a in basic_atoms]),
                        np.array([a.position for a in ligand_acidic])
                    )
                    if np.min(d) <= SALTBRIDGE_CUTOFF:
                        present = True

            if present:
                counts[(resid, res.resname)] = counts.get((resid, res.resname), 0) + 1

    rows = []
    for (resid, resname), count in counts.items():
        rows.append({
            "resid": resid,
            "resname": resname,
            "interaction_type": "salt_bridge",
            "persistence": count / n_frames
        })

    return pd.DataFrame(rows)

# ============================================================
# HYDROPHOBIC CONTACTS
# ============================================================

def compute_hydrophobic_persistence(u, ligand_resid, pocket_resids, stride=10):
    ligand = u.select_atoms(f"resid {ligand_resid}")
    ligand_hydrophobic = ligand.select_atoms("not name H*")

    ligand_hydrophobic = [a for a in ligand_hydrophobic if atom_element(a) in HYDROPHOBIC_ELEMENTS]

    counts = {}
    n_frames = 0

    pocket_groups = {}
    for resid in pocket_resids:
        grp = u.select_atoms(f"protein and resid {resid} and not name H*")
        hyd = [a for a in grp if atom_element(a) in HYDROPHOBIC_ELEMENTS]
        if hyd:
            pocket_groups[resid] = hyd

    for ts in u.trajectory[::stride]:
        n_frames += 1

        if not ligand_hydrophobic:
            break

        lig_pos = np.array([a.position for a in ligand_hydrophobic])

        for resid, atoms in pocket_groups.items():
            resname = atoms[0].residue.resname
            prot_pos = np.array([a.position for a in atoms])
            d = distance_array(prot_pos, lig_pos)
            if np.min(d) <= HYDROPHOBIC_CUTOFF:
                counts[(resid, resname)] = counts.get((resid, resname), 0) + 1

    rows = []
    for (resid, resname), count in counts.items():
        rows.append({
            "resid": resid,
            "resname": resname,
            "interaction_type": "hydrophobic",
            "persistence": count / n_frames if n_frames > 0 else np.nan
        })

    return pd.DataFrame(rows)

# ============================================================
# PI-PI STACKING
# ============================================================

def compute_pipi_persistence(u, ligand_resid, pocket_resids, stride=10):
    ligand = u.select_atoms(f"resid {ligand_resid} and not name H*")
    ligand_ring_indices = detect_ligand_rings(ligand)

    if not ligand_ring_indices:
        return pd.DataFrame(columns=["resid", "resname", "interaction_type", "persistence"])

    pocket_residue_objs = [u.select_atoms(f"protein and resid {r}").residues[0] for r in pocket_resids]
    counts = {}
    n_frames = 0

    for ts in u.trajectory[::stride]:
        n_frames += 1

        # ligand rings
        ligand_rings = []
        for ring_idx in ligand_ring_indices:
            ring_atoms = ligand.atoms[ring_idx]
            coords = ring_atoms.positions
            centroid, normal = ring_centroid_and_normal(coords)
            ligand_rings.append((centroid, normal))

        for res in pocket_residue_objs:
            prings = get_protein_ring_atoms(res)
            if not prings:
                continue

            resid = int(res.resid)
            resname = res.resname
            present = False

            for pr in prings:
                pcoords = np.array([a.position for a in pr])
                pcent, pnorm = ring_centroid_and_normal(pcoords)

                for lcent, lnorm in ligand_rings:
                    d = np.linalg.norm(pcent - lcent)
                    if d > PI_PI_CENTROID_CUTOFF:
                        continue

                    ang = angle_deg(pnorm, lnorm)
                    if ang is None:
                        continue

                    # face-to-face or T-shaped
                    parallel_like = ang <= PI_PI_PLANE_ANGLE_CUTOFF
                    tshape_like = abs(ang - PI_T_ANGLE_TARGET) <= PI_T_ANGLE_TOL

                    if parallel_like or tshape_like:
                        present = True
                        break
                if present:
                    break

            if present:
                counts[(resid, resname)] = counts.get((resid, resname), 0) + 1

    rows = []
    for (resid, resname), count in counts.items():
        rows.append({
            "resid": resid,
            "resname": resname,
            "interaction_type": "pi_pi",
            "persistence": count / n_frames if n_frames > 0 else np.nan
        })

    return pd.DataFrame(rows)

# ============================================================
# RUN ONE SYSTEM
# ============================================================

def analyze_system(sysinfo):
    print(f"Analyzing {sysinfo['protein']} / {sysinfo['ligand_name']}")

    outdir = os.path.join(OUTDIR, f"{sysinfo['protein']}_{sysinfo['ligand_key']}")
    os.makedirs(outdir, exist_ok=True)

    u = mda.Universe(sysinfo["prmtop"], sysinfo["trajs"])
    u = prepare_trajectory(u)

    pocket_info = define_pocket_residues(u, sysinfo["ligand_resid"], cutoff=POCKET_CUTOFF)
    pocket_resids = [r for r, _ in pocket_info]

    # individual interaction fingerprints
    hbond_df = compute_hbond_persistence(u, sysinfo["ligand_resid"], pocket_resids, stride=STRIDE)

    # rewind by reloading universe for analyses that iterate trajectory
    u = mda.Universe(sysinfo["prmtop"], sysinfo["trajs"])
    u = prepare_trajectory(u)
    salt_df = compute_saltbridge_persistence(u, sysinfo["ligand_resid"], pocket_resids, stride=STRIDE)

    u = mda.Universe(sysinfo["prmtop"], sysinfo["trajs"])
    u = prepare_trajectory(u)
    hyd_df = compute_hydrophobic_persistence(u, sysinfo["ligand_resid"], pocket_resids, stride=STRIDE)

    u = mda.Universe(sysinfo["prmtop"], sysinfo["trajs"])
    u = prepare_trajectory(u)
    pipi_df = compute_pipi_persistence(u, sysinfo["ligand_resid"], pocket_resids, stride=STRIDE)

    all_df = pd.concat([hbond_df, salt_df, hyd_df, pipi_df], ignore_index=True)
    all_df["protein"] = sysinfo["protein"]
    all_df["ligand"] = sysinfo["ligand_name"]

    all_df.to_csv(os.path.join(outdir, "interaction_persistence.csv"), index=False)

    # summary by interaction type
    summary = (
        all_df.groupby("interaction_type")
        .agg(
            n_residues=("resid", "nunique"),
            mean_persistence=("persistence", "mean"),
            max_persistence=("persistence", "max"),
        )
        .reset_index()
    )
    summary.to_csv(os.path.join(outdir, "interaction_summary.csv"), index=False)

    # top residues overall
    top_df = all_df.sort_values(["interaction_type", "persistence"], ascending=[True, False])
    top_df.to_csv(os.path.join(outdir, "interaction_top_residues.csv"), index=False)

    with open(os.path.join(outdir, "pocket_residues.json"), "w") as f:
        json.dump(
            [{"resid": int(r), "resname": str(rn)} for r, rn in pocket_info],
            f,
            indent=2
        )

    return all_df

# ============================================================
# MAIN
# ============================================================

def main():
    os.makedirs(OUTDIR, exist_ok=True)

    systems = discover_systems()
    if not systems:
        raise FileNotFoundError("No systems found.")

    all_rows = []

    for sysinfo in systems:
        try:
            df = analyze_system(sysinfo)
            all_rows.append(df)
        except Exception as e:
            print(f"Failed for {sysinfo['protein']} / {sysinfo['ligand_name']}: {e}")

    if all_rows:
        final_df = pd.concat(all_rows, ignore_index=True)
        final_df.to_csv(os.path.join(OUTDIR, "interaction_persistence_all_systems.csv"), index=False)

        summary_all = (
            final_df.groupby(["protein", "ligand", "interaction_type"])
            .agg(
                n_residues=("resid", "nunique"),
                mean_persistence=("persistence", "mean"),
                max_persistence=("persistence", "max"),
            )
            .reset_index()
        )
        summary_all.to_csv(os.path.join(OUTDIR, "interaction_summary_all_systems.csv"), index=False)

        print("\nInteraction fingerprint summary\n")
        print(summary_all.to_string(index=False))

if __name__ == "__main__":
    main()
