"""Planner Agent — decomposes the brief into a typed TaskPlan."""

from __future__ import annotations

from datetime import datetime, timezone

from ..schemas import EvidenceCard, TaskNode, TaskPlan
from ..state import AgentState


def _initial_plan(brief: str) -> TaskPlan:
    nodes = [
        TaskNode(
            name="research_target",
            agent="research",
            success_criteria="TargetBrief with UniProt + ≥3 PDB + ≥5 literature evidences",
            tools_allowed=["uniprot", "pdb", "literature", "websearch"],
            estimated_cost={"wallclock_s": 240, "llm_tokens": 30000},
        ),
        TaskNode(
            name="critic_target_brief",
            agent="critic",
            success_criteria="Every claim in TargetBrief has ≥1 EvidenceCard",
            estimated_cost={"wallclock_s": 30},
        ),
        TaskNode(
            name="structure_epitope_map",
            agent="structure",
            success_criteria="EpitopeMap with ≥5 hotspots, each consensus_score ≥ 2",
            tools_allowed=["pdb_fetch", "biotite", "interface_analysis", "conservation"],
            estimated_cost={"wallclock_s": 180},
        ),
        TaskNode(
            name="critic_epitope",
            agent="critic",
            success_criteria="Cross-tool agreement on hotspots ≥ 2 sources",
            estimated_cost={"wallclock_s": 30},
        ),
        TaskNode(
            name="design_candidates",
            agent="design",
            success_criteria="≥20 linear + ≥2 cyclic candidates with design_provenance",
            tools_allowed=["mutation_scan", "esm_if", "boltz_protein_design", "rfdiffusion"],
            estimated_cost={"wallclock_s": 1800, "gpu_h": 0.5},
        ),
        TaskNode(
            name="critic_candidates",
            agent="critic",
            success_criteria="Provenance + rationale evidence-gated for every candidate",
            estimated_cost={"wallclock_s": 60},
        ),
        TaskNode(
            name="predict_complexes",
            agent="prediction",
            success_criteria="ScoreCard per candidate; Boltz API + Chai-1 on top-5",
            tools_allowed=["boltz_api", "chai_1", "interface_analysis"],
            estimated_cost={"wallclock_s": 3600, "gpu_h": 0.2, "usd": 30},
        ),
        TaskNode(
            name="critic_scores",
            agent="critic",
            success_criteria="Cross-tool agreement + calibrated rejection applied",
            estimated_cost={"wallclock_s": 60},
        ),
        TaskNode(
            name="report",
            agent="reporter",
            success_criteria="Markdown + provenance appendix; top candidates + failure log",
            estimated_cost={"wallclock_s": 60},
        ),
    ]
    return TaskPlan(brief=brief, nodes=nodes)


def run(state: AgentState) -> dict:
    plan = state.get("plan")
    if plan is None:
        plan = _initial_plan(state["brief"])
        # Drop an evidence card for the decision itself
        card = EvidenceCard(
            claim=f"Planner emitted initial plan with {len(plan.nodes)} nodes",
            source_id=plan.plan_id,
            source_type="agent_decision",
            tag="DERIVED",
            confidence=1.0,
            extracted_by="planner",
            payload={"plan_id": plan.plan_id, "node_names": [n.name for n in plan.nodes]},
        )
        return {
            "plan": plan,
            "evidence_ledger": [card],
            "history": [
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "agent": "planner",
                    "kind": "decision",
                    "payload": {"action": "initial_plan", "n_nodes": len(plan.nodes)},
                }
            ],
        }
    # Re-planning path (after Critic veto): minimal v0 — just bump revisions
    plan.revisions += 1
    return {"plan": plan}
