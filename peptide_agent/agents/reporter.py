# v2-RECONSTRUCTED from transcript+spec (worker termination 2026-06-27)
# Verbatim sources:
#   - L340/L693 (v1 baseline, identical)
#   - L766 Edit verbatim (inserts ## Critic convergence log section)
#   - L769 ExecuteCode (pre-fix rendered outputs for C9/C10/C11)
#   - L788 Edit args (pre-fix banner code, replaced by _format_convergence_status)
#   - L791 ExecuteCode (post-fix rendered outputs for C10-redux/C9/C11)
#   - step5_smoke_results.json: 14 unit tests + 8 routing + E2E confirming behavior
"""Reporter Agent — render the final user-facing report.

Produces:
  - report.md: full Markdown with inline citations
  - provenance_appendix.md: every claim → EvidenceCard hash → source URL

# v2 ADDITIONS:
#   - _format_convergence_status(state): trajectory-aware status banner + per-agent
#     critic convergence log table. Looks at the full critic_reports history (not
#     the live critic_iterations counter, which resets to 0 on a pass), so a
#     pipeline that vetoed once and recovered shows "completed after 1 veto and
#     self-correction" rather than misleadingly claiming a clean first pass.
#   - "Gated failure" callout: when halt_reason == "critic_unconvergent", the
#     report opens its convergence section with an explicit warning that the
#     candidate set is the LAST ATTEMPTED generation, not a validated shortlist.
#   - Banner inserted into Executive summary after the critic_reports count line.
#   - L766 Edit verbatim: "## Failure log" header changed to
#     "## Critic convergence log" + "## Failure log and Critic vetoes" split.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from ..schemas import EvidenceCard
from ..state import AgentState


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ----------------------------------------------------------------------
# Section formatters (v1, preserved verbatim from L340/L693)
# ----------------------------------------------------------------------


def _format_candidate_table(state: AgentState) -> str:
    rows = []
    rows.append("| Rank | Cand ID | Sequence | Modality | Length | ipTM | hotspot_cov | Class | Strategy |")
    rows.append("|---:|---|---|---|---:|---:|---:|---|---|")

    scores = state.get("scores", {})
    candidates = {c.cand_id: c for c in state.get("candidates", [])}
    ranked = sorted(scores.values(), key=lambda s: -s.composite_score)
    for i, s in enumerate(ranked, 1):
        c = candidates.get(s.cand_id)
        if not c:
            continue
        ip = s.structural.get("mean_ipTM", "")
        cov = s.interface.get("hotspot_coverage_fraction", "")
        rows.append(f"| {i} | `{c.cand_id}` | `{c.sequence}` | {c.modality} | {c.length} | "
                    f"{ip if ip == '' else f'{ip:.3f}'} | {cov if cov == '' else f'{cov:.0%}'} | "
                    f"{s.confidence_class} | {c.design_provenance.generator} |")
    if len(rows) == 2:
        rows.append("| — | (no predictions completed yet) | | | | | | | |")
    return "\n".join(rows)


def _format_hotspots(state: AgentState) -> str:
    emap = state.get("epitope_map")
    if not emap:
        return "(no epitope map)"
    rows = ["| Residue | Role | Consensus | Tools |", "|---|---|---:|---|"]
    for h in sorted(emap.hotspots, key=lambda x: (-x.consensus_score, x.residue_number)):
        rows.append(f"| {h.residue_aa}{h.residue_number} (chain {h.chain}) | {h.role} | {h.consensus_score} | {', '.join(h.supported_by_tools)} |")
    return "\n".join(rows)


def _format_failure_log(state: AgentState) -> str:
    items = []
    for c in state.get("candidates", []):
        if c.status in ("filtered_out", "rejected"):
            items.append(f"- `{c.cand_id}` ({c.sequence}, {c.modality}, status={c.status}) — {c.filter_reason or 'see Critic reports'}")
    for r in state.get("critic_reports", []):
        if r.verdict in ("warn", "veto"):
            for iss in r.issues:
                items.append(f"- Critic({r.target_agent}): {iss.severity.upper()} [{iss.layer}] — {iss.message}")
    return "\n".join(items) if items else "(no failures recorded)"


def _format_evidence_appendix(state: AgentState) -> str:
    ledger = state.get("evidence_ledger", [])
    lines = ["| card_id | hash | tag | claim | source |", "|---|---|---|---|---|"]
    for c in ledger:
        url = c.source_url or c.source_id
        claim = (c.claim[:140] + "…") if len(c.claim) > 140 else c.claim
        lines.append(f"| `{c.card_id}` | `{c.content_hash or '—'}` | {c.tag} | {claim} | {url} |")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# v2 trajectory-aware convergence section
# ----------------------------------------------------------------------


# [v2 NEW] L788 Edit args (pre-fix banner, INTENTIONALLY replaced) — kept here
# in a comment to document the bug fix:
#
#   pre-fix (BUG):
#     if halt_reason == "critic_unconvergent":
#         banner = (f"- **Status: HALTED — pipeline did not converge** "
#                   f"(Critic iter cap reached at {critic_iters} consecutive vetoes).")
#     elif halt_reason:
#         banner = f"- **Status: HALTED** ({halt_reason}, after {critic_iters} Critic iterations)."
#     elif critic_iters > 0:
#         banner = f"- **Status: completed after {critic_iters} Critic loop(s).**"
#     else:
#         banner = "- **Status: completed on first pass** (no Critic vetoes)."
#
# The bug: critic_iters resets to 0 on a successful pass (per critic.py iteration
# bookkeeping). C10 (veto then pass) ended with critic_iters=0, so the banner
# misleadingly said "first pass". The v2 fix counts vetoes from the full
# critic_reports history instead.


def _format_convergence_status(state: AgentState) -> tuple[str, str]:
    """Render the trajectory-aware banner + convergence log table.

    Returns (banner_line, log_section). Banner goes in Executive summary;
    log_section becomes the body under "## Critic convergence log".

    # Behavior verified against step5 smoke tests:
    #   - C9 (clean):       "completed on first pass (no Critic vetoes)."
    #   - C10-redux (1 veto then pass):
    #         "completed after 1 Critic veto(s) and self-correction (design×1)."
    #   - C11 (halt):       "HALTED — pipeline did not converge
    #                        (Critic iter cap reached at 3 consecutive vetoes on design)."
    """
    reports = state.get("critic_reports", []) or []
    halt_reason = state.get("halt_reason")

    # Count vetoes per target_agent across the full history.
    per_agent_stats: dict[str, dict] = defaultdict(
        lambda: {"reports": 0, "passes": 0, "warns": 0, "vetoes": 0,
                 "last_verdict": None, "last_hint": None}
    )
    for r in reports:
        s = per_agent_stats[r.target_agent]
        s["reports"] += 1
        if r.verdict == "pass":
            s["passes"] += 1
        elif r.verdict == "warn":
            s["warns"] += 1
        elif r.verdict == "veto":
            s["vetoes"] += 1
        s["last_verdict"] = r.verdict
        s["last_hint"] = (r.recommended_action or "").strip() or None

    total_vetoes = sum(s["vetoes"] for s in per_agent_stats.values())

    # ---- Banner -----------------------------------------------------------
    if halt_reason == "critic_unconvergent":
        # The agent that finally halted = the last veto target.
        last_veto_agent = None
        for r in reversed(reports):
            if r.verdict == "veto":
                last_veto_agent = r.target_agent
                break
        # critic_iterations holds the actual consecutive-veto count at halt time.
        iters_at_halt = int(state.get("critic_iterations", 0))
        if last_veto_agent:
            banner = (f"- **Status: HALTED — pipeline did not converge** "
                      f"(Critic iter cap reached at {iters_at_halt} consecutive vetoes on {last_veto_agent}).")
        else:
            banner = (f"- **Status: HALTED — pipeline did not converge** "
                      f"(Critic iter cap reached at {iters_at_halt} consecutive vetoes).")
    elif halt_reason:
        banner = f"- **Status: HALTED** ({halt_reason})."
    elif total_vetoes > 0:
        # Compose "(design×1, prediction×2)" style agent×count summary.
        veto_breakdown = ", ".join(
            f"{ag}×{stats['vetoes']}"
            for ag, stats in per_agent_stats.items() if stats["vetoes"] > 0
        )
        banner = (f"- **Status: completed after {total_vetoes} Critic veto(s) and self-correction** "
                  f"({veto_breakdown}).")
    else:
        banner = "- **Status: completed on first pass** (no Critic vetoes)."

    # ---- Convergence log table -------------------------------------------
    lines: list[str] = []
    if halt_reason == "critic_unconvergent":
        iters_at_halt = int(state.get("critic_iterations", 0))
        lines.append(
            f"> **Gated failure**: the Critic vetoed the same upstream agent for "
            f"{iters_at_halt} consecutive iterations without convergence. The "
            f"candidate set below reflects the **last attempted** generation, not "
            f"a validated final shortlist."
        )
        lines.append("")

    lines.append("| Critic target | Reports | Passes | Warns | Vetoes | Last verdict | Last replan hint |")
    lines.append("|---|---:|---:|---:|---:|---|---|")

    # Always render all 4 agents in canonical pipeline order so empty cycles
    # show up as "(not invoked)" rather than disappearing silently.
    for agent in ("research", "structure", "design", "prediction"):
        s = per_agent_stats.get(agent)
        if s is None or s["reports"] == 0:
            lines.append(f"| {agent} | 0 | — | — | — | (not invoked) | — |")
        else:
            verdict_cell = f"`{s['last_verdict']}`"
            hint_cell = s["last_hint"] or "—"
            lines.append(
                f"| {agent} | {s['reports']} | {s['passes']} | {s['warns']} | "
                f"{s['vetoes']} | {verdict_cell} | {hint_cell} |"
            )

    return banner, "\n".join(lines)


# ----------------------------------------------------------------------
# Main render
# ----------------------------------------------------------------------


def run(state: AgentState) -> dict:
    out_dir = Path(state["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "report.md"

    brief = state.get("target_brief")
    emap = state.get("epitope_map")
    n_cands = sum(1 for c in state.get("candidates", []) if c.status not in ("filtered_out",))
    n_scored = len(state.get("scores", {}))
    n_evid = len(state.get("evidence_ledger", []))
    n_critic = len(state.get("critic_reports", []))

    # [v2 NEW] Render trajectory-aware status BEFORE building the markdown so we
    # can inject the banner into Executive summary.
    status_banner, convergence_log = _format_convergence_status(state)

    md = []
    md.append(f"# PeptideAgent-v0 Report — {brief.target_id if brief else state['target_id']}")
    md.append(f"\n**Run**: `{state['run_id']}` &nbsp;&nbsp; **Generated**: {_now()}")
    md.append("\n## Executive summary")
    md.append(f"- Target: **{brief.target_id if brief else state['target_id']}** "
              f"(UniProt {brief.uniprot if brief else '?'}, gene {brief.gene if brief else '?'}, "
              f"{brief.length if brief else '?'} aa).")
    md.append(f"- Proposed candidates: **{n_cands}** ({sum(1 for c in state.get('candidates', []) if c.modality == 'linear')} linear, "
              f"{sum(1 for c in state.get('candidates', []) if c.modality.startswith('cyclic'))} cyclic).")
    md.append(f"- Scored complexes: **{n_scored}**.")
    md.append(f"- EvidenceCards committed: **{n_evid}**.")
    md.append(f"- Critic reports: **{n_critic}** "
              f"(vetoes={sum(1 for r in state.get('critic_reports', []) if r.verdict=='veto')}, "
              f"warnings={sum(1 for r in state.get('critic_reports', []) if r.verdict=='warn')}).")
    # [v2 NEW] inject trajectory-aware status banner here
    md.append(status_banner)

    md.append("\n## Target brief")
    if brief:
        md.append(brief.function_summary)
        if brief.interaction_partners:
            md.append(f"\n**Known interaction partners**: {', '.join(brief.interaction_partners)}.")
        if brief.known_binders:
            md.append("\n**Reported binders/inhibitors**:")
            for b in brief.known_binders:
                md.append(f"- {b['name']} — {b.get('modality','?')} ({b.get('source','?')})")
        md.append(f"\n**Reference PDBs**: {', '.join(brief.reference_pdbs)}.")
    else:
        md.append("(no TargetBrief produced — Critic should have vetoed)")

    md.append("\n## Epitope map")
    if emap:
        md.append(emap.summary)
        md.append("")
        md.append(_format_hotspots(state))
    else:
        md.append("(no epitope map)")

    md.append("\n## Top candidates")
    md.append(_format_candidate_table(state))

    md.append("\n## Cross-tool validation")
    if state.get("scores"):
        chai_used = any(
            "chai" in str(s.consistency).lower() for s in state["scores"].values()
        )
        md.append(f"- Boltz-2.1 ensemble: yes")
        md.append(f"- Chai-1 cross-check on top-5: {'yes' if chai_used else 'deferred — see Critic warnings'}")
    else:
        md.append("(no scores yet)")

    # [v2 NEW: L766 Edit verbatim] Convergence log section replaces the bare
    # "## Failure log" header from v1.
    md.append("\n## Critic convergence log")
    md.append(convergence_log)

    md.append("\n## Failure log and Critic vetoes")
    md.append(_format_failure_log(state))

    md.append("\n## Next-step experiments (proposed)")
    md.append("- **SPR binding**: top 5 candidates against recombinant human PD-L1 ECD (Q9NZQ7 19–238); confirm ipTM-predicted ranking.")
    md.append("- **Cell-based PD-L1 blockade assay** (T-cell activation): top 3 candidates after SPR triage.")
    md.append("- **CD spectroscopy + thermal melt**: secondary-structure stability for cyclic candidates.")
    md.append("- **Selectivity counterscreen**: PD-L2 (UniProt Q9BQ51) to confirm PD-L1 specificity.")

    md.append("\n---\n")
    md.append("## Provenance appendix")
    md.append("Every claim above traces back to one of the following EvidenceCards. Hashes are deterministic over `(claim, source_id, source_type, extracted_by, payload, derived_from)`.\n")
    md.append(_format_evidence_appendix(state))

    report_path.write_text("\n".join(md))

    return {
        "report_path": str(report_path),
        "history": [{"timestamp": _now(), "agent": "reporter", "kind": "decision",
                     "payload": {"report_path": str(report_path)}}],
    }
