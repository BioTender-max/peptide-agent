"""Mammalian orthologue conservation for proteins.

Genuinely independent signal from PDB contact analysis: fetches UniProt orthologs
across mammals, pairwise-aligns each to the reference human sequence, and returns
per-position conservation fractions (count of orthologs matching human residue,
normalised by total orthologs aligned).

For PD-L1 / CD274 specifically, the canonical CD274 orthologs across ~9 mammals
are well-characterised; for other targets we fall back to a UniProt search by gene.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from dataclasses import dataclass
from functools import lru_cache

import biotite.sequence as seq_lib
import biotite.sequence.align as align_lib
import numpy as np


# Mammal organism IDs for ortholog search
MAMMAL_TAXON_IDS = [9606, 10090, 10116, 9544, 9598, 9615, 9913, 9823, 9986]


@dataclass
class ConservationResult:
    reference_acc: str
    reference_seq: str
    orthologs: list[dict]  # {organism, acc, seq, identity, source}
    conservation: list[float]  # per-position fraction, 0..1, len = len(reference_seq)


@lru_cache(maxsize=8)
def _search_uniprot_orthologs(gene: str, taxon_ids: tuple[int, ...]) -> tuple:
    """Search UniProt for entries matching gene name across given taxa."""
    tax_clause = " OR ".join(f"organism_id:{t}" for t in taxon_ids)
    query = f"(gene:{gene}) AND ({tax_clause})"
    url = "https://rest.uniprot.org/uniprotkb/search?" + urllib.parse.urlencode({
        "query": query,
        "format": "json",
        "fields": "accession,organism_name,length,sequence,reviewed",
        "size": 50,
    })
    with urllib.request.urlopen(url, timeout=30) as r:
        data = json.load(r)
    out = []
    for entry in data["results"]:
        out.append({
            "acc": entry["primaryAccession"],
            "org": entry["organism"]["scientificName"],
            "seq": entry["sequence"]["value"],
            "reviewed": entry.get("entryType", "?"),
        })
    return tuple(json.dumps(o, sort_keys=True) for o in out)  # cache-friendly


def fetch_orthologs(gene: str, reference_acc: str, taxon_ids: list[int] | None = None) -> list[dict]:
    """Fetch orthologs and pick best one per organism (reviewed preferred, sane length)."""
    taxon_ids = taxon_ids or MAMMAL_TAXON_IDS
    results = _search_uniprot_orthologs(gene, tuple(sorted(taxon_ids)))
    entries = [json.loads(s) for s in results]

    # Get reference sequence to filter by length sanity
    ref_entry = next((e for e in entries if e["acc"] == reference_acc), None)
    if ref_entry is None:
        # fetch reference directly
        url = f"https://rest.uniprot.org/uniprotkb/{reference_acc}.fasta"
        with urllib.request.urlopen(url, timeout=15) as r:
            text = r.read().decode()
        ref_seq = "".join(text.strip().split("\n")[1:])
    else:
        ref_seq = ref_entry["seq"]
    ref_len = len(ref_seq)

    # Pick best entry per organism: prefer reviewed, then length within ±15% of reference
    by_org = {}
    for e in entries:
        if e["acc"] == reference_acc:
            continue
        if abs(len(e["seq"]) - ref_len) / ref_len > 0.20:
            continue  # skip orthologs that are too short/long (likely truncated/spliced)
        prev = by_org.get(e["org"])
        if prev is None:
            by_org[e["org"]] = e
            continue
        # Prefer reviewed; if both unreviewed, prefer one closer to ref length
        prev_reviewed = "Reviewed" in (prev.get("reviewed", "") or "")
        e_reviewed = "Reviewed" in (e.get("reviewed", "") or "")
        if e_reviewed and not prev_reviewed:
            by_org[e["org"]] = e
        elif not e_reviewed and not prev_reviewed:
            if abs(len(e["seq"]) - ref_len) < abs(len(prev["seq"]) - ref_len):
                by_org[e["org"]] = e
    return list(by_org.values())


def compute_conservation(gene: str, reference_acc: str, reference_seq: str | None = None) -> ConservationResult:
    """Align orthologs to reference and return per-position conservation.

    For each position i in reference_seq, conservation[i] = fraction of orthologs
    whose aligned residue equals reference_seq[i].
    """
    if reference_seq is None:
        url = f"https://rest.uniprot.org/uniprotkb/{reference_acc}.fasta"
        with urllib.request.urlopen(url, timeout=15) as r:
            text = r.read().decode()
        reference_seq = "".join(text.strip().split("\n")[1:])

    orthologs = fetch_orthologs(gene, reference_acc)
    if not orthologs:
        return ConservationResult(reference_acc=reference_acc, reference_seq=reference_seq,
                                  orthologs=[], conservation=[0.0] * len(reference_seq))

    matrix = align_lib.SubstitutionMatrix.std_protein_matrix()
    human_bio = seq_lib.ProteinSequence(reference_seq)

    match_count = np.zeros(len(reference_seq), dtype=int)
    aligned_count = 0
    annotated_orthologs = []
    for o in orthologs:
        try:
            other = seq_lib.ProteinSequence(o["seq"])
        except Exception:
            continue
        alns = align_lib.align_optimal(human_bio, other, matrix,
                                       gap_penalty=(-10, -1), terminal_penalty=False)
        if not alns:
            continue
        aln = alns[0]
        identical = 0
        for h_idx, o_idx in aln.trace:
            if h_idx == -1 or o_idx == -1:
                continue
            if reference_seq[h_idx] == o["seq"][o_idx]:
                match_count[h_idx] += 1
                identical += 1
        annotated_orthologs.append({
            "acc": o["acc"],
            "org": o["org"],
            "seq_len": len(o["seq"]),
            "identity_to_reference": identical / len(reference_seq),
            "source": "uniprot",
        })
        aligned_count += 1

    if aligned_count == 0:
        conservation = [0.0] * len(reference_seq)
    else:
        conservation = (match_count / aligned_count).tolist()

    return ConservationResult(
        reference_acc=reference_acc,
        reference_seq=reference_seq,
        orthologs=annotated_orthologs,
        conservation=conservation,
    )
