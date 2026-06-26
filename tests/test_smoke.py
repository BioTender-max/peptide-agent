"""Smoke tests — exercise the scaffold without external API calls.

Run from /mnt/results/peptide-agent:
    PYTHONPATH=. python -m tests.test_smoke
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

# Allow running as `python -m tests.test_smoke`
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_imports() -> None:
    from peptide_agent import schemas, state, ledger, graph
    from peptide_agent.agents import planner, research, structure, design, prediction, critic, reporter, supervisor
    from peptide_agent.tools import uniprot, pdb, literature, interface, boltz_api
    print("[smoke] imports OK")


def test_schemas_construct() -> None:
    from peptide_agent.schemas import (
        EvidenceCard, TaskNode, TaskPlan, TargetBrief, Hotspot, EpitopeMap,
        Candidate, DesignProvenance, ComplexPrediction, ScoreCard, CriticReport, Issue,
    )
    e = EvidenceCard(
        claim="Test claim",
        source_id="test_source",
        source_type="agent_decision",
        tag="DERIVED",
        confidence=0.5,
        extracted_by="smoke_test",
        payload={"k": "v"},
    )
    assert e.card_id.startswith("evid_")
    n = TaskNode(name="t", agent="planner", success_criteria="ok")
    p = TaskPlan(brief="design a binder for PD-L1", nodes=[n])
    assert len(p.nodes) == 1
    assert p.plan_id.startswith("plan_")
    h = Hotspot(chain="A", residue_number=56, residue_aa="Y", role="anchor", consensus_score=3)
    em = EpitopeMap(target_id="PD-L1", reference_pdb="5O45", hotspots=[h], summary="x")
    prov = DesignProvenance(generator="mutation_scan", parent_sequence="NYSKPTDRQYHF")
    c = Candidate(sequence="NYSKPTDRQYHF", modality="linear", length=12,
                  design_provenance=prov, design_rationale="x")
    s = ScoreCard(cand_id=c.cand_id, structural={"mean_ipTM": 0.62},
                  interface={"hotspot_coverage_fraction": 0.7},
                  composite_score=0.7, confidence_class="medium")
    assert s.composite_score > 0
    print("[smoke] schemas construct OK")


def test_ledger_roundtrip() -> None:
    from peptide_agent.schemas import EvidenceCard
    from peptide_agent.ledger.store import EvidenceLedger

    tmp = Path(tempfile.mkdtemp(prefix="peptide_smoke_"))
    try:
        ledger = EvidenceLedger(tmp / "ledger.jsonl")
        c1 = EvidenceCard(claim="A", source_id="s1", source_type="uniprot",
                          tag="VERIFIED", extracted_by="t", payload={"x": 1})
        c2 = EvidenceCard(claim="A", source_id="s1", source_type="uniprot",
                          tag="VERIFIED", extracted_by="t", payload={"x": 1})  # duplicate of c1
        c3 = EvidenceCard(claim="B", source_id="s2", source_type="pdb",
                          tag="VERIFIED", extracted_by="t", payload={"y": 2})
        added = ledger.append(c1)
        assert added.content_hash is not None
        dup = ledger.append(c2)
        assert dup.card_id == added.card_id, f"expected dedup; got {dup.card_id} vs {added.card_id}"
        ledger.append(c3)

        # Reload from disk
        reload = EvidenceLedger(tmp / "ledger.jsonl")
        active = reload.get_active()
        assert len(active) == 2, f"expected 2 active; got {len(active)}"
        by_type = reload.by_source_type("uniprot")
        assert len(by_type) == 1, f"expected 1 uniprot card; got {len(by_type)}"
        by_src = reload.by_source("s1")
        assert len(by_src) == 1, f"expected 1 card with source_id='s1'; got {len(by_src)}"
        print("[smoke] ledger dedup + reload OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_graph_compiles() -> None:
    from peptide_agent.graph import build_graph, initial_state
    g = build_graph(checkpoint=False)
    s = initial_state(run_id="smoke_run", target_id="PD-L1",
                      brief="Design peptide binders for PD-L1",
                      out_dir="/tmp/peptide_smoke")
    assert s["target_id"] == "PD-L1"
    assert s["plan"] is None
    print("[smoke] graph compiles OK")


def test_planner_produces_plan() -> None:
    from peptide_agent.agents import planner
    from peptide_agent.graph import initial_state
    s = initial_state(run_id="smoke_run", target_id="PD-L1",
                      brief="Design peptide binders for PD-L1",
                      out_dir="/tmp/peptide_smoke")
    upd = planner.run(s)
    assert upd["plan"] is not None
    n = len(upd["plan"].nodes)
    assert n >= 5, f"expected ≥5 nodes; got {n}"
    assert len(upd["evidence_ledger"]) == 1
    print(f"[smoke] planner produced {n}-node plan OK")


def test_supervisor_routing() -> None:
    from peptide_agent.agents.supervisor import next_node
    from peptide_agent.graph import initial_state
    s = initial_state(run_id="smoke_run", target_id="PD-L1",
                      brief="Design", out_dir="/tmp/peptide_smoke")
    assert next_node(s) == "planner"
    print("[smoke] supervisor initial routing OK")


def main() -> int:
    tests = [
        test_imports,
        test_schemas_construct,
        test_ledger_roundtrip,
        test_graph_compiles,
        test_planner_produces_plan,
        test_supervisor_routing,
    ]
    failed = []
    for t in tests:
        try:
            t()
        except Exception as exc:
            failed.append((t.__name__, exc, traceback.format_exc()))
            print(f"[smoke] FAIL: {t.__name__}: {exc}")
    if failed:
        for name, exc, tb in failed:
            print(f"\n=== FAILED: {name} ===\n{tb}")
        return 1
    print(f"\n[smoke] ALL {len(tests)} TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
