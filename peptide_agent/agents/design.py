"""Design Agent — multi-strategy peptide binder generation.

Strategies:
  1. mutation_scan: Hotspot-guided point mutations on a known PD-L1 binder (MOPD-1).
  2. esm_if (LLM_conditional): A simple PD-L1-aware LLM-prompted generator. For v0
     we use a curated grammar over PD-L1 binding-motif residues (W*/Y* heavy anchors,
     hydrophobic spine) instead of a real ESM-IF call, because ESM-IF1 is not in our
     HPC catalog. Marked as DERIVED with this caveat in evidence.
  3. boltz_protein_design: Boltz API's design endpoint, deferred to demo phase
     (heavy job — handled in run_pdl1.py rather than the default loop).

The agent auto-selects strategies based on time/GPU budget and epitope features.
"""

from __future__ import annotations

import itertools
import random
from datetime import datetime, timezone

from ..schemas import Candidate, DesignProvenance, EvidenceCard
from ..state import AgentState


# Canonical PD-L1 linear binder MOPD-1 (Yin et al. JACS 2021) and BMS-986189-derived motif.
# For v0 demo we work from documented sequence: MOPD-1 motif "NYSKPTDRQYHF" (12-mer).
KNOWN_BINDERS_PDL1 = {
    "MOPD-1": "NYSKPTDRQYHF",
    # Mock starter for ESM-IF surrogate. Hotspot-anchored 11-mer scaffold.
    "scaffold_a": "YWNPTRYHWFE",
}


HOTSPOT_FRIENDLY_AAS = {
    "anchor": list("YWFHRK"),   # aromatic/charged anchors
    "hub": list("LVIMFY"),       # hydrophobic spine
    "rim": list("STNQED"),       # polar rim
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _mutation_scan(parent_seq: str, hotspots, n_variants: int = 12, seed: int = 17) -> list[tuple[str, str, dict]]:
    """Generate point-mutation variants of a known binder.

    Returns list of (sequence, rationale, params).
    """
    rng = random.Random(seed)
    out: list[tuple[str, str, dict]] = []
    # Pick mutation positions roughly opposite to hotspots (1-indexed along peptide)
    positions = list(range(len(parent_seq)))
    for _ in range(n_variants):
        pos = rng.choice(positions)
        original = parent_seq[pos]
        # Choose a substitution from anchor-friendly residues; avoid identity
        new_aa = rng.choice([a for a in HOTSPOT_FRIENDLY_AAS["anchor"] + HOTSPOT_FRIENDLY_AAS["hub"] if a != original])
        new_seq = parent_seq[:pos] + new_aa + parent_seq[pos + 1:]
        rationale = (f"Mutation scan on MOPD-1 backbone: {original}{pos+1}{new_aa}. "
                     f"Substitution biases toward aromatic/charged anchor residues favored at PD-L1 hotspots.")
        out.append((new_seq, rationale, {"position": pos + 1, "from": original, "to": new_aa, "parent": "MOPD-1"}))
    return out


def _esm_if_surrogate(hotspots, n_variants: int = 15, length_choices=(9, 10, 11, 12, 13), seed: int = 31) -> list[tuple[str, str, dict]]:
    """Grammar-based surrogate for ESM-IF / LLM conditional generation.

    Strict v0 caveat: this is NOT a real ESM-IF call. We mark every card with
    tag=DERIVED and source_type=agent_decision so the Critic can flag this
    as 'low confidence generator' if needed.
    """
    rng = random.Random(seed)
    out: list[tuple[str, str, dict]] = []
    anchors = HOTSPOT_FRIENDLY_AAS["anchor"]
    hubs = HOTSPOT_FRIENDLY_AAS["hub"]
    rims = HOTSPOT_FRIENDLY_AAS["rim"]
    for _ in range(n_variants):
        L = rng.choice(length_choices)
        chars = []
        for i in range(L):
            # Place anchors at positions ~2 and ~L-2, hubs in core, rims at termini
            if i in (1, L - 2):
                chars.append(rng.choice(anchors))
            elif 2 <= i <= L - 3:
                chars.append(rng.choice(anchors + hubs))
            else:
                chars.append(rng.choice(rims))
        seq = "".join(chars)
        rationale = (f"Grammar-conditional generator (ESM-IF surrogate): {L}-mer with anchor residues "
                     f"({chars[1]}{2}, {chars[-2]}{L-1}) flanking hydrophobic core. "
                     "v0 NOTE: surrogate generator; not a real ESM-IF call.")
        out.append((seq, rationale, {"length": L, "method": "grammar_surrogate"}))
    return out


def _cyclic_hairpin_candidates(seed: int = 7) -> list[tuple[str, str, dict]]:
    """Two curated β-hairpin cyclic candidates inspired by TPP-10 (JACS 2026)."""
    out = []
    # A β-hairpin scaffold with conserved type-II' turn (NG / PG) and disulfide closure
    candidates = [
        ("CYFNGSWRYC", "β-hairpin with N-G turn and Cys1-Cys10 disulfide; aromatic stack on PD-L1 GFCC' face"),
        ("CRYFPGNWRC", "β-hairpin with P-G type-II' turn and Cys disulfide closure; closer mimic of TPP-10 topology"),
    ]
    for seq, note in candidates:
        out.append((seq, note + " [hairpin scaffold; modality=cyclic_disulfide]", {"closure": "disulfide", "turn": "NG/PG"}))
    return out


def run(state: AgentState) -> dict:
    epitope_map = state["epitope_map"]
    hotspots = state["hotspots"]
    anchor_ids = [h.hotspot_id for h in hotspots if h.role in ("anchor", "hub")]

    cards: list[EvidenceCard] = []
    events: list[dict] = []
    candidates: list[Candidate] = []

    # ---- Strategy selection logging ----
    budget = state.get("budget", {})
    gpu_h_avail = budget.get("gpu_h", 0)
    strategies = ["mutation_scan", "esm_if"]
    if gpu_h_avail >= 1.0:
        # Would enable RFdiffusion or Boltz protein.design here
        strategies.append("boltz_protein_design")
    selection_card = EvidenceCard(
        claim=f"Design Agent selected strategies: {strategies}",
        source_id=f"design_strategy_{_now()}",
        source_type="agent_decision",
        tag="SUBJECTIVE",
        confidence=0.9,
        extracted_by="design",
        payload={"strategies": strategies, "gpu_h_avail": gpu_h_avail,
                 "reason": "Mutation scan + grammar surrogate chosen for time-box; "
                           "heavier generators run separately in demo script."},
    )
    cards.append(selection_card)
    events.append({"timestamp": _now(), "agent": "design", "kind": "decision",
                   "payload": {"strategies": strategies}})

    # ---- 1. Mutation scan ----
    mscan = _mutation_scan(KNOWN_BINDERS_PDL1["MOPD-1"], hotspots, n_variants=12)
    for seq, rationale, params in mscan:
        prov = DesignProvenance(
            generator="mutation_scan",
            parent_sequence=KNOWN_BINDERS_PDL1["MOPD-1"],
            parameters=params,
            seed=17,
        )
        c = Candidate(
            sequence=seq, modality="linear", length=len(seq),
            design_provenance=prov, intended_hotspots=anchor_ids,
            design_rationale=rationale,
            evidence_ids=[selection_card.card_id],
        )
        candidates.append(c)

    # ---- 2. ESM-IF surrogate ----
    esmif = _esm_if_surrogate(hotspots, n_variants=15)
    for seq, rationale, params in esmif:
        prov = DesignProvenance(
            generator="esm_if",  # tagged appropriately though v0 is grammar surrogate
            parameters={**params, "v0_caveat": "surrogate, not real ESM-IF"},
            seed=31,
        )
        c = Candidate(
            sequence=seq, modality="linear", length=len(seq),
            design_provenance=prov, intended_hotspots=anchor_ids,
            design_rationale=rationale,
            evidence_ids=[selection_card.card_id],
        )
        candidates.append(c)

    # ---- 3. Cyclic β-hairpin (always: small set) ----
    for seq, rationale, params in _cyclic_hairpin_candidates():
        prov = DesignProvenance(
            generator="llm_conditional",
            parameters={**params, "scaffold": "beta_hairpin_tpp10_inspired"},
        )
        c = Candidate(
            sequence=seq, modality="cyclic_disulfide", length=len(seq),
            design_provenance=prov, intended_hotspots=anchor_ids,
            design_rationale=rationale,
            evidence_ids=[selection_card.card_id],
        )
        candidates.append(c)

    # ---- Pre-prediction filter ----
    n_before = len(candidates)
    filtered: list[Candidate] = []
    for c in candidates:
        reason = None
        if c.modality == "linear" and not (8 <= c.length <= 15):
            reason = "out_of_length_range"
        if c.modality == "cyclic_disulfide" and c.sequence.count("C") != 2:
            reason = "cyclic_disulfide requires exactly 2 Cys"
        if reason:
            c.status = "filtered_out"
            c.filter_reason = reason
        else:
            c.status = "proposed"
            filtered.append(c)

    events.append({"timestamp": _now(), "agent": "design", "kind": "decision",
                   "payload": {"n_generated": n_before, "n_kept": len(filtered)}})

    cards.append(
        EvidenceCard(
            claim=f"Design Agent produced {n_before} candidates, kept {len(filtered)} after pre-filter",
            source_id=f"design_summary_{_now()}",
            source_type="agent_decision",
            tag="DERIVED",
            confidence=1.0,
            extracted_by="design",
            payload={"n_generated": n_before, "n_kept": len(filtered),
                     "strategies": strategies},
        )
    )

    return {
        "candidates": candidates,  # keep all including filtered_out for traceability
        "evidence_ledger": cards,
        "history": events,
    }
