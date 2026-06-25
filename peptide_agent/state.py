"""LangGraph AgentState — the working memory of one run."""

from __future__ import annotations

from typing import Annotated, Optional, TypedDict
from operator import add

from .schemas import (
    Candidate,
    ComplexPrediction,
    CriticReport,
    EpitopeMap,
    EvidenceCard,
    Hotspot,
    ScoreCard,
    StructureProfile,
    TargetBrief,
    TaskPlan,
)


class Event(TypedDict, total=False):
    """One entry in the append-only event log."""
    timestamp: str
    agent: str
    kind: str          # tool_call, tool_return, decision, critic_verdict
    payload: dict


class AgentState(TypedDict, total=False):
    # Identity
    run_id: str
    target_id: str
    brief: str
    out_dir: str       # /mnt/results/runs/<run_id>

    # Planning
    plan: Optional[TaskPlan]
    current_task: Optional[str]
    history: Annotated[list[Event], add]

    # Knowledge artifacts
    target_brief: Optional[TargetBrief]
    structure_profile: Optional[StructureProfile]
    evidence_ledger: Annotated[list[EvidenceCard], add]
    hotspots: list[Hotspot]
    epitope_map: Optional[EpitopeMap]

    # Design
    candidates: list[Candidate]
    predictions: dict[str, list[ComplexPrediction]]   # cand_id -> [ComplexPrediction]
    scores: dict[str, ScoreCard]                      # cand_id -> ScoreCard

    # Quality control
    critic_reports: Annotated[list[CriticReport], add]
    veto_queue: list[dict]
    confidence_floor: float
    critic_iterations: int       # v2: counts veto-triggered replans, capped at 2
    last_critic_target: Optional[str]  # which agent the last veto rerouted to
    critic_feedback: Optional[str]     # message passed from Critic to the replanned agent
    halt_reason: Optional[str]   # set if iter cap reached or unrecoverable error

    # Output
    report_path: Optional[str]

    # Cost & budget
    budget: dict        # remaining $$, gpu_hours, wallclock_s
    spent: dict
