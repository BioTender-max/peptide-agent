"""LangGraph composition.

build_graph() returns a compiled StateGraph that takes a brief and produces
a full AgentState with a rendered report.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver

from .agents import critic, design, planner, prediction, reporter, research, structure
from .state import AgentState


def _critic_router_factory(target_agent: str):
    def _node(state: AgentState) -> dict:
        return critic.critique(state, target_agent=target_agent)
    return _node


CRITIC_ITER_CAP = 2


def _route_after_critic(state: AgentState) -> str:
    """Decide where to go after a Critic pass.

    v2 semantics (PLAN.md §2.2):
      - verdict == "pass" or "warn" → continue to next agent in pipeline
      - verdict == "veto" AND critic_iterations < cap → loop back to the agent that can fix it
      - verdict == "veto" AND critic_iterations >= cap → "halt" (Reporter renders gated-failure)
    """
    reports = state.get("critic_reports") or []
    if not reports:
        return "planner"
    last = reports[-1]

    # Standard forward map
    forward = {
        "research":   "structure",
        "structure":  "design",
        "design":     "prediction",
        "prediction": "reporter",
    }
    # On veto, which upstream agent should we re-run?
    veto_back = {
        "research":   "research",
        "structure":  "structure",
        "design":     "design",
        # Prediction veto = data is fine, the candidates are bad → bounce to Design
        "prediction": "design",
    }

    if last.verdict == "veto":
        iters = int(state.get("critic_iterations", 0))
        if iters >= CRITIC_ITER_CAP:
            return "reporter"  # halt path: Reporter renders the gated-failure report
        return veto_back.get(last.target_agent, "reporter")

    return forward.get(last.target_agent, "reporter")


def build_graph(checkpoint: bool = True):
    g = StateGraph(AgentState)

    g.add_node("planner",          planner.run)
    g.add_node("research",         research.run)
    g.add_node("structure",        structure.run)
    g.add_node("design",           design.run)
    g.add_node("prediction",       prediction.run)
    g.add_node("reporter",         reporter.run)

    g.add_node("critic_research",   _critic_router_factory("research"))
    g.add_node("critic_structure",  _critic_router_factory("structure"))
    g.add_node("critic_design",     _critic_router_factory("design"))
    g.add_node("critic_prediction", _critic_router_factory("prediction"))

    # v2 topology: Critic veto loops back to the upstream agent (capped at 2 iterations).
    g.add_edge(START, "planner")
    g.add_edge("planner", "research")
    g.add_edge("research", "critic_research")
    g.add_conditional_edges("critic_research", _route_after_critic,
                             {"structure": "structure", "research": "research",
                              "reporter": "reporter"})
    g.add_edge("structure", "critic_structure")
    g.add_conditional_edges("critic_structure", _route_after_critic,
                             {"design": "design", "structure": "structure",
                              "reporter": "reporter"})
    g.add_edge("design", "critic_design")
    g.add_conditional_edges("critic_design", _route_after_critic,
                             {"prediction": "prediction", "design": "design",
                              "reporter": "reporter"})
    g.add_edge("prediction", "critic_prediction")
    g.add_conditional_edges("critic_prediction", _route_after_critic,
                             {"reporter": "reporter", "design": "design"})
    g.add_edge("reporter", END)

    if checkpoint:
        return g.compile(checkpointer=MemorySaver())
    return g.compile()


def initial_state(run_id: str, target_id: str, brief: str, out_dir: str) -> AgentState:
    return AgentState(
        run_id=run_id,
        target_id=target_id,
        brief=brief,
        out_dir=out_dir,
        plan=None,
        current_task=None,
        history=[],
        target_brief=None,
        structure_profile=None,
        evidence_ledger=[],
        hotspots=[],
        epitope_map=None,
        candidates=[],
        predictions={},
        scores={},
        critic_reports=[],
        veto_queue=[],
        confidence_floor=0.5,
        critic_iterations=0,
        last_critic_target=None,
        critic_feedback=None,
        halt_reason=None,
        report_path=None,
        budget={"usd": 50.0, "gpu_h": 2.0, "wallclock_s": 7200},
        spent={},
    )
