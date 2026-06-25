"""Structure Agent — load reference PDB, compute interface, mammalian conservation, hotspots.

Cross-tool agreement uses two genuinely independent signals:
  Axis A: structural contact (PDB heavy-atom <5 Å in the cocrystal)
  Axis B: mammalian conservation (≥ 7/8 orthologs match human residue)

A residue is labelled:
  - "anchor"   : in contact AND highly conserved (both axes)  → consensus=2
  - "rim"      : in contact AND variable                       → consensus=1

This is deliberately less ambitious than a 3-axis cross-tool score, but every
axis is independently computed from primary data — no curated hot-spot lists.
A future v1 can add genuine mutagenesis-extracted literature as a third axis.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

from ..schemas import EpitopeMap, EvidenceCard, Hotspot, StructureProfile
from ..state import AgentState
from ..tools.conservation import compute_conservation
from ..tools.interface import compute_interface_contacts
from ..tools.pdb import fetch_pdb_file


CONS_THRESHOLD = 0.875  # ≥ 7/8 orthologs identical to human residue


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_or_compute_conservation(
    gene: str,
    reference_acc: str,
    reference_seq: str,
    cache_dir: str | None,
) -> dict[str, Any]:
    """Try cache first; else compute via tool. Returns dict with conservation list + ortholog metadata."""
    if cache_dir:
        cache_file = os.path.join(cache_dir, f"conservation_{reference_acc.lower()}.json")
        if os.path.exists(cache_file):
            with open(cache_file) as f:
                cached = json.load(f)
            if cached.get("reference_seq") == reference_seq:
                return cached

    result = compute_conservation(gene, reference_acc, reference_seq=reference_seq)
    payload = {
        "reference_acc": result.reference_acc,
        "reference_gene": gene,
        "reference_seq": result.reference_seq,
        "reference_length": len(result.reference_seq),
        "n_orthologs": len(result.orthologs),
        "orthologs": result.orthologs,
        "conservation": [round(float(c), 4) for c in result.conservation],
        "method": "Biotite pairwise alignment, BLOSUM62, gap=(-10,-1), terminal_penalty=False",
    }
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"conservation_{reference_acc.lower()}.json")
        with open(cache_file, "w") as f:
            json.dump(payload, f, indent=2)
    return payload


def run(state: AgentState) -> dict:
    target_brief = state["target_brief"]
    pdbs = target_brief.reference_pdbs
    if not pdbs:
        raise RuntimeError("Structure Agent requires reference_pdbs in TargetBrief")

    cards: list[EvidenceCard] = []
    events: list[dict] = []

    # Reference PDB must be a true cocrystal (≥2 chains). Research agent orders them.
    reference_pdb = pdbs[0]
    pdb_file = fetch_pdb_file(reference_pdb)
    events.append({"timestamp": _now(), "agent": "structure", "kind": "tool_call",
                   "payload": {"tool": "pdb_fetch", "id": reference_pdb}})

    # ---- Axis A: structural contact ----
    interface = compute_interface_contacts(pdb_file, target_chain="auto", partner_chain="auto")
    target_chain = interface["target_chain"]
    partner_chain = interface["partner_chain"]
    target_residues = interface["target_residues"]
    events.append({"timestamp": _now(), "agent": "structure", "kind": "tool_call",
                   "payload": {"tool": "interface_analysis",
                               "n_target_residues_at_interface": len(target_residues),
                               "n_atom_contacts_total": interface["n_contacts_total"]}})

    cards.append(
        EvidenceCard(
            claim=f"{target_brief.target_id} chain {target_chain} interfaces with partner chain {partner_chain} "
                  f"in PDB {reference_pdb}; {len(target_residues)} target residues within 5Å of partner heavy atoms",
            source_id=f"{reference_pdb}_interface",
            source_type="tool_output",
            tag="DERIVED",
            confidence=0.95,
            extracted_by="structure",
            payload={"reference_pdb": reference_pdb,
                     "n_target_residues": len(target_residues),
                     "target_chain": target_chain,
                     "partner_chain": partner_chain},
        )
    )

    # ---- Axis B: mammalian orthologue conservation ----
    # Reference sequence: prefer the one captured on the TargetBrief; else require user to provide.
    reference_seq = target_brief.sequence
    reference_acc = target_brief.uniprot
    gene = target_brief.gene
    cache_dir = getattr(target_brief, "cache_dir", None) or "/mnt/results/peptide-agent/cache"

    conservation_data = None
    conservation = None
    if reference_seq and reference_acc and gene:
        try:
            conservation_data = _load_or_compute_conservation(
                gene=gene,
                reference_acc=reference_acc,
                reference_seq=reference_seq,
                cache_dir=cache_dir,
            )
            conservation = conservation_data["conservation"]
            events.append({"timestamp": _now(), "agent": "structure", "kind": "tool_call",
                           "payload": {"tool": "compute_conservation",
                                       "n_orthologs": conservation_data["n_orthologs"],
                                       "mean_conservation": round(sum(conservation) / len(conservation), 3)}})
            cards.append(
                EvidenceCard(
                    claim=f"Conservation of {target_brief.target_id} ({reference_acc}) computed from "
                          f"{conservation_data['n_orthologs']} mammalian orthologs; mean conservation "
                          f"{sum(conservation) / len(conservation):.2f}",
                    source_id=f"conservation_{reference_acc}",
                    source_type="tool_output",
                    tag="DERIVED",
                    confidence=0.9,
                    extracted_by="structure",
                    payload={"n_orthologs": conservation_data["n_orthologs"],
                             "orthologs": [o["acc"] for o in conservation_data["orthologs"]],
                             "method": conservation_data["method"]},
                )
            )
        except Exception as exc:
            events.append({"timestamp": _now(), "agent": "structure", "kind": "tool_error",
                           "payload": {"tool": "compute_conservation", "error": str(exc)}})

    # ---- Build hotspot list with 2-axis consensus ----
    hotspots: list[Hotspot] = []
    for r in target_residues:
        resi = r["resi"]
        in_contact = True
        cons_value = None
        in_conservation = False
        if conservation and 0 <= (resi - 1) < len(conservation):
            cons_value = conservation[resi - 1]
            in_conservation = cons_value >= CONS_THRESHOLD

        tools_supporting = ["pdb_contact"]
        if in_conservation:
            tools_supporting.append("mammalian_conservation")

        consensus = len(tools_supporting)
        if in_contact and in_conservation:
            role = "anchor"
        else:
            role = "rim"

        hs = Hotspot(
            chain=r["chain"],
            residue_number=resi,
            residue_aa=r["resn"],
            role=role,
            bsa=r.get("bsa"),
            conservation=cons_value,
            supported_by_tools=tools_supporting,
            consensus_score=consensus,
        )
        hotspots.append(hs)

        cons_str = f", conservation={cons_value:.2f}" if cons_value is not None else ""
        cards.append(
            EvidenceCard(
                claim=f"{target_brief.target_id} residue {r['resn']}{resi} on chain {r['chain']} "
                      f"is a {role} (consensus={consensus}{cons_str})",
                source_id=f"{reference_pdb}_res_{r['chain']}_{resi}",
                source_type="tool_output",
                tag="DERIVED",
                confidence=min(0.99, 0.7 + 0.1 * consensus),
                extracted_by="structure",
                payload={"role": role,
                         "tools": tools_supporting,
                         "bsa": r.get("bsa"),
                         "conservation": cons_value},
            )
        )

    # Attach evidence_ids back to hotspots
    for card in cards:
        for h in hotspots:
            tag = f"_res_{h.chain}_{h.residue_number}"
            if tag in card.source_id:
                h.evidence_ids.append(card.card_id)

    # ---- Build EpitopeMap ----
    n_anchor = sum(1 for h in hotspots if h.role == "anchor")
    n_rim = sum(1 for h in hotspots if h.role == "rim")
    if conservation is None:
        cons_note = "no conservation data (UniProt accession/gene missing)"
    else:
        cons_note = f"using {conservation_data['n_orthologs']} mammalian orthologs at ≥{CONS_THRESHOLD:.0%} threshold"

    summary = (f"{target_brief.target_id} interface on chain {target_chain}: "
               f"{len(hotspots)} residues within 5 Å of chain {partner_chain} in PDB {reference_pdb}; "
               f"{n_anchor} anchors (contact + conserved), {n_rim} rim (contact only). "
               f"Cross-tool agreement {cons_note}.")
    epitope_map = EpitopeMap(
        target_id=target_brief.target_id,
        reference_pdb=reference_pdb,
        partner_chain=partner_chain,
        hotspots=hotspots,
        summary=summary,
        evidence_ids=[c.card_id for c in cards],
    )

    profile = StructureProfile(
        target_id=target_brief.target_id,
        pdbs_loaded=[reference_pdb],
        chains={reference_pdb: interface["chain_sizes"]},
        epitope_map=epitope_map,
        notes=(f"Interface from {reference_pdb} chain {target_chain} vs {partner_chain} (cutoff 5 Å); "
               f"conservation from {conservation_data['n_orthologs'] if conservation_data else 0} orthologs. "
               f"v0 cross-tool axes: {{pdb_contact, mammalian_conservation}}. "
               "No curated literature hotspot list used (avoids confirmation bias)."),
    )

    return {
        "structure_profile": profile,
        "epitope_map": epitope_map,
        "hotspots": hotspots,
        "evidence_ledger": cards,
        "history": events,
    }
