"""Structure interface analysis using Biotite.

Identifies the target chain, the partner chain, and computes per-residue
contacts within a heavy-atom distance cutoff.

For PD-1 / PD-L1 PDB 5C3T the canonical chains are A (PD-1) and B (PD-L1).
Auto-detection picks the two chains with the most inter-chain contacts.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx


def _load_structure(cif_path: str) -> struc.AtomArray:
    cif = pdbx.CIFFile.read(cif_path)
    arr = pdbx.get_structure(cif, model=1)
    # Restrict to protein heavy atoms
    arr = arr[(arr.hetero == False) & (arr.element != "H")]
    return arr


def _chain_pair_contacts(arr: struc.AtomArray, cutoff: float = 5.0) -> tuple[str, str, int]:
    chains = np.unique(arr.chain_id).tolist()
    if len(chains) < 2:
        raise ValueError(
            f"Structure has only {len(chains)} polymer chain(s): {chains}. "
            "Interface analysis requires a complex with ≥2 chains. "
            "Use a cocrystal PDB (e.g. 4ZQK or 3BIK for PD-1/PD-L1) instead of a monomer."
        )
    best = (None, None, -1)
    for i in range(len(chains)):
        for j in range(i + 1, len(chains)):
            a = arr[arr.chain_id == chains[i]]
            b = arr[arr.chain_id == chains[j]]
            if len(a) == 0 or len(b) == 0:
                continue
            # Use the all-vs-all coordinate broadcast (struc.distance with repeat/tile
            # returns shape mismatches on some Biotite versions). Direct numpy is safer.
            diffs = a.coord[:, None, :] - b.coord[None, :, :]
            dist = np.sqrt((diffs ** 2).sum(axis=-1))
            n_contacts = int(np.sum(dist <= cutoff))
            if n_contacts > best[2]:
                best = (chains[i], chains[j], n_contacts)
    if best[0] is None:
        raise ValueError(
            f"No chain pair has heavy-atom contacts within {cutoff} Å. "
            f"Chains were: {chains}"
        )
    return best


def compute_interface_contacts(
    cif_path: str,
    target_chain: str = "auto",
    partner_chain: str = "auto",
    cutoff: float = 5.0,
) -> dict:
    arr = _load_structure(cif_path)

    if target_chain == "auto" or partner_chain == "auto":
        c1, c2, _ = _chain_pair_contacts(arr, cutoff=cutoff)
        # Decide which chain is the "target":
        #   For PD-L1 work (5C3T) we prefer the larger PD-L1 chain (B).
        a = arr[arr.chain_id == c1]
        b = arr[arr.chain_id == c2]
        if len(np.unique(b.res_id)) >= len(np.unique(a.res_id)):
            target_chain, partner_chain = c2, c1
        else:
            target_chain, partner_chain = c1, c2

    tgt = arr[arr.chain_id == target_chain]
    part = arr[arr.chain_id == partner_chain]
    # All-vs-all atom distances
    tgt_coord = tgt.coord
    part_coord = part.coord
    # Compute per-target-atom min-distance to any partner atom
    diffs = tgt_coord[:, None, :] - part_coord[None, :, :]
    dists = np.sqrt((diffs ** 2).sum(axis=-1))
    min_per_atom = dists.min(axis=1)

    # Aggregate to residue
    res_min: dict[tuple[str, int], tuple[str, float]] = {}
    for atom_idx, (chain, resi, resn, d) in enumerate(zip(tgt.chain_id, tgt.res_id, tgt.res_name, min_per_atom)):
        key = (chain, int(resi))
        if key not in res_min or d < res_min[key][1]:
            res_min[key] = (resn, float(d))

    target_residues = [
        {"chain": chain, "resi": resi, "resn": resn,
         "min_dist_to_partner": dist, "bsa": None}
        for (chain, resi), (resn, dist) in res_min.items()
        if dist <= cutoff
    ]
    target_residues.sort(key=lambda x: x["resi"])

    chain_sizes = {c: int(len(np.unique(arr[arr.chain_id == c].res_id))) for c in np.unique(arr.chain_id)}

    return {
        "target_chain": target_chain,
        "partner_chain": partner_chain,
        "target_residues": target_residues,
        "n_contacts_total": int(np.sum(dists <= cutoff)),
        "chain_sizes": chain_sizes,
        "cif_path": cif_path,
    }


def score_interface_from_structure(cif_path: str, hotspot_resis: list[int]) -> dict:
    """Given a predicted complex CIF, compute hotspot coverage."""
    arr = _load_structure(cif_path)
    chains = np.unique(arr.chain_id).tolist()
    # Target chain = longer; peptide = shorter
    sizes = {c: len(np.unique(arr[arr.chain_id == c].res_id)) for c in chains}
    target_chain = max(sizes, key=sizes.get)
    peptide_chain = min(sizes, key=sizes.get)
    interface = compute_interface_contacts(cif_path, target_chain=target_chain, partner_chain=peptide_chain)
    contacted = {r["resi"] for r in interface["target_residues"]}
    covered = contacted & set(hotspot_resis)
    coverage_fraction = len(covered) / max(1, len(hotspot_resis))
    return {
        "contacted_target_residues": sorted(contacted),
        "hotspots_covered": sorted(covered),
        "hotspot_coverage_fraction": coverage_fraction,
        "interface_residue_count": len(contacted),
    }
