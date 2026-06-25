"""Supervisor Agent — LangGraph conditional router.

The Supervisor inspects state and decides which agent runs next.
For v0 we use a deterministic sequence; Critic vetoes route back to Planner.
"""

from __future__ import annotations

from ..state import AgentState


def next_node(state: AgentState) -> str:
    # Latest critic verdict, if any, gates progression
    crit_reports = state.get("critic_reports") or []
    last_critic = crit_reports[-1] if crit_reports else None

    if last_critic and last_critic.verdict == "veto":
        return "planner"  # request replan

    has_plan = state.get("plan") is not None
    has_brief = state.get("target_brief") is not None
    has_epitope = state.get("epitope_map") is not None
    has_candidates = bool(state.get("candidates"))
    has_predictions = bool(state.get("predictions"))
    has_scores = bool(state.get("scores"))
    has_report = state.get("report_path") is not None

    # If the supervisor reached us after an agent ran, decide next based on what is missing.
    if not has_plan:
        return "planner"

    # Check what was the last agent run via critic_reports to alternate agent → critic
    last_agent_run = state.get("history", [{}])[-1].get("agent") if state.get("history") else None

    # Insert critic after each substantive agent
    if last_agent_run in ("research",) and (not last_critic or last_critic.target_agent != "research"):
        return "critic_research"
    if last_agent_run in ("structure",) and (not last_critic or last_critic.target_agent != "structure"):
        return "critic_structure"
    if last_agent_run in ("design",) and (not last_critic or last_critic.target_agent != "design"):
        return "critic_design"
    if last_agent_run in ("prediction",) and (not last_critic or last_critic.target_agent != "prediction"):
        return "critic_prediction"

    if not has_brief:
        return "research"
    if not has_epitope:
        return "structure"
    if not has_candidates:
        return "design"
    if not has_predictions:
        return "prediction"
    if not has_scores:
        # Prediction submitted but scores not assembled yet — call reporter with partial state
        # (real demo: wait for callbacks, then call assemble_scores_from_completed)
        return "reporter"
    if not has_report:
        return "reporter"
    return "END"
