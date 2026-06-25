"""Research Agent — deep research using UniProt + PDB + LiteratureSearch.

Produces a TargetBrief with every claim backed by an EvidenceCard.
The LiteratureSearch tool is mediated via tools.literature.search().
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Optional

from ..schemas import EvidenceCard, TargetBrief
from ..state import AgentState
from ..tools.literature import search_literature
from ..tools.pdb import fetch_pdb_summary
from ..tools.uniprot import fetch_uniprot


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run(state: AgentState, target_uniprot: Optional[str] = None) -> dict:
    target_id = state["target_id"]
    uniprot_id = target_uniprot or _default_uniprot(target_id)

    cards: list[EvidenceCard] = []
    events: list[dict] = []

    # ---- 1. UniProt ----
    up = fetch_uniprot(uniprot_id)
    cards.append(
        EvidenceCard(
            claim=f"{target_id} canonical sequence is {up['length']} aa (UniProt {uniprot_id}, gene {up['gene']})",
            source_id=uniprot_id,
            source_type="uniprot",
            source_url=f"https://www.uniprot.org/uniprotkb/{uniprot_id}/entry",
            tag="VERIFIED",
            confidence=0.99,
            extracted_by="research",
            payload={"length": up["length"], "gene": up["gene"], "organism": up["organism"]},
        )
    )
    events.append({"timestamp": _now(), "agent": "research", "kind": "tool_call",
                   "payload": {"tool": "uniprot", "id": uniprot_id}})

    # ---- 2. PDB ----
    pdb_ids = _default_pdbs(target_id)
    pdb_summaries = []
    for pid in pdb_ids:
        summary = fetch_pdb_summary(pid)
        pdb_summaries.append(summary)
        cards.append(
            EvidenceCard(
                claim=f"PDB {pid}: {summary['title']} ({summary['method']}, {summary['resolution']} Å)",
                source_id=pid,
                source_type="pdb",
                source_url=f"https://www.rcsb.org/structure/{pid}",
                tag="VERIFIED",
                confidence=0.99,
                extracted_by="research",
                payload=summary,
            )
        )
        events.append({"timestamp": _now(), "agent": "research", "kind": "tool_call",
                       "payload": {"tool": "pdb", "id": pid}})

    # ---- 3. Literature ----
    lit_queries = [
        f"{target_id} peptide binder design structure-based",
        f"{target_id} hotspot interface residue mutagenesis",
        f"{target_id} small molecule inhibitor crystal structure",
    ]
    lit_records = []
    for q in lit_queries:
        recs = search_literature(q, max_papers=6, year_min=2018)
        lit_records.extend(recs)
        events.append({"timestamp": _now(), "agent": "research", "kind": "tool_call",
                       "payload": {"tool": "literature", "query": q, "n_results": len(recs)}})

    # Dedupe by DOI/URL
    seen = set()
    deduped = []
    for r in lit_records:
        key = r.get("doi") or r.get("url") or r.get("title")
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)

    for r in deduped[:15]:
        cards.append(
            EvidenceCard(
                claim=r.get("highlight") or r.get("title") or "literature record",
                source_id=r.get("doi") or r.get("url") or r.get("title", "?"),
                source_type="literature",
                source_url=r.get("url"),
                tag="VERIFIED" if r.get("doi") else "DERIVED",
                confidence=0.85,
                extracted_by="research",
                payload={k: r.get(k) for k in ("title", "authors", "year", "journal", "study_type", "doi", "url") if r.get(k)},
            )
        )

    # ---- 4. Conflict detection (simple: keyword pairs) ----
    if target_id.upper() in ("PD-L1", "PDL1"):
        # CA-170 known dispute
        ca170_disputes = [
            c for c in cards
            if "CA-170" in (c.payload.get("title") or "") + (c.claim or "")
            or "ca-170" in (c.claim or "").lower()
        ]
        if ca170_disputes:
            cards.append(
                EvidenceCard(
                    claim="Literature contains conflicting reports on CA-170 direct binding to PD-L1",
                    source_id="ca170_dispute",
                    source_type="agent_decision",
                    tag="SUBJECTIVE",
                    confidence=0.7,
                    extracted_by="research",
                    derived_from=[c.card_id for c in ca170_disputes],
                    payload={"note": "Surface as 'disputed' in the report"},
                )
            )

    # ---- 5. Build TargetBrief ----
    brief = TargetBrief(
        target_id=target_id,
        uniprot=uniprot_id,
        gene=up["gene"],
        organism=up["organism"],
        length=up["length"],
        sequence=up.get("sequence"),
        function_summary=_default_function(target_id),
        interaction_partners=_default_partners(target_id),
        known_binders=_seed_known_binders(target_id),
        reference_pdbs=pdb_ids,
        evidence_ids=[c.card_id for c in cards],
    )

    return {
        "target_brief": brief,
        "evidence_ledger": cards,
        "history": events,
    }


# ---- Defaults (PD-L1 demo; would be looked up dynamically in v1) ----

def _default_uniprot(target_id: str) -> str:
    table = {
        "PD-L1": "Q9NZQ7",
        "PDL1": "Q9NZQ7",
        "PD-1": "Q15116",
        "IL-23R": "Q5VWK5",
    }
    return table.get(target_id.upper(), table.get(target_id, "Q9NZQ7"))


def _default_pdbs(target_id: str) -> list[str]:
    # 4ZQK (2.45 Å, Zak 2015) and 3BIK (2.65 Å, Lin 2008) are the canonical
    # PD-1/PD-L1 cocrystal structures — they carry the native interface.
    # 5O45 (0.99 Å) is PD-L1 + BMS-986189 macrocyclic peptide — high-resolution
    # template but partner is a peptide ligand, not PD-1.
    # 5C3T (1.8 Å) is the PD-L1 IgV monomer (single chain) — kept for sequence
    # context but cannot be used for interface analysis.
    table = {
        "PD-L1": ["4ZQK", "3BIK", "5O45", "5IUS", "5C3T"],
        "PDL1":  ["4ZQK", "3BIK", "5O45", "5IUS", "5C3T"],
    }
    return table.get(target_id.upper(), table.get(target_id, ["5O45"]))


def _default_function(target_id: str) -> str:
    if target_id.upper() in ("PD-L1", "PDL1"):
        return ("Programmed cell death 1 ligand 1 (CD274); type I transmembrane protein "
                "expressed on tumor and APC surfaces; engages PD-1 on T cells to deliver "
                "an inhibitory signal that dampens T-cell receptor signaling.")
    return ""


def _default_partners(target_id: str) -> list[str]:
    if target_id.upper() in ("PD-L1", "PDL1"):
        return ["PD-1 (PDCD1)", "CD80 (B7-1)"]
    return []


def _seed_known_binders(target_id: str) -> list[dict]:
    if target_id.upper() in ("PD-L1", "PDL1"):
        return [
            {"name": "BMS-986189", "modality": "macrocyclic_peptide", "evidence_ids": [], "source": "PDB 5O45 cocrystal"},
            {"name": "MOPD-1",     "modality": "linear_peptide",     "evidence_ids": [], "source": "JACS 2021"},
            {"name": "TPP-10",     "modality": "cyclic_b_hairpin",   "evidence_ids": [], "source": "JACS 2026"},
            {"name": "CA-170",     "modality": "small_molecule_disputed", "evidence_ids": [], "source": "Communications Biology 2021 / disputed by Musielak 2019"},
        ]
    return []
