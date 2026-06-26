# v2-RECONSTRUCTED from transcript+spec (worker termination 2026-06-27)
# Verbatim sources:
#   - L338 (full v1 baseline, 7,003 chars)
#   - L747 ExecuteCode dump (imports + tunables + _evidence_gate_artifact start)
#   - L749 grep (line-number map of every `if` condition; 38 lines)
#   - L751 Read lines 274-308 (full target_agent dispatch verbatim)
#   - L753 Read lines 110-139 (_cross_tool_hotspots full + _cross_tool_predictions start)
#   - L745 inspect.getsource (L112=_cross_tool_hotspots, L131=_cross_tool_predictions)
#   - L741 ExecuteCode (anchor pattern format: ".AAAAAAAA." for 10-mer with anchors @ 2-9)
# Spec sources:
#   - critic_config.json (tunables, anchor_residues, dispatch_order, iteration_bookkeeping)
#   - step5_smoke_results.json (C1-C8 test expectations: trigger_layer + verdict per scenario)
"""Critic Agent — 4-layer hallucination control + iteration bookkeeping.

# v2 ADDITIONS over v1:
#   Layer 1 (evidence_gate): split into _evidence_gate_artifact (any artifact
#     with evidence_ids), _evidence_gate_hotspots (hotspots without
#     supported_by_tools), and _evidence_gate_epitope_map (reference_pdb must
#     be cited by ≥1 ledger card).
#   Layer 2 (cross_tool): _cross_tool_hotspots (consensus_score >= 2),
#     _cross_tool_predictions (top-K Boltz vs Chai-1 mean |Δ ipTM| > 0.15 → veto).
#   Layer 3 (self_consistency): _self_consistency_predictions (per-cand ipTM
#     stdev across seeds > 0.10 → warn), _diversity_design (<5 distinct anchor
#     patterns across ≥5 candidates → veto with diversity replan hint).
#   Layer 4 (calibrated_rejection): _calibrated_rejection_predictions (≥80%
#     candidates in "rejected" class → veto), _calibrated_rejection_design
#     (≥80% filtered_out before scoring → veto).
#
# Iteration bookkeeping (NEW; per critic_config.json):
#   - veto:  critic_iterations += 1; critic_feedback = recommended_action;
#            if new_iters > 2 → halt_reason = "critic_unconvergent"
#   - pass:  critic_iterations = 0; critic_feedback = None
#   - warn:  no change
#
# State channels touched on EVERY pass:
#   - critic_reports (append)
#   - evidence_ledger (append; self-critique card)
#   - history (append)
#   - critic_iterations (overwrite via "<channel>": value, NOT add)
#   - critic_feedback (overwrite)
#   - last_critic_target (overwrite — the target_agent name)
#   - halt_reason: set when the iter cap has already been hit and we still veto
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from statistics import mean, pstdev
from typing import Iterable

from ..schemas import CriticReport, EvidenceCard, Issue
from ..state import AgentState

# --- Tunables (kept readable; PLAN values applied here) ---------------
# [L747 verbatim]
DISAGREEMENT_THRESHOLD = 0.15      # mean |boltz_ipTM - chai_ipTM| on top-5
SELF_CONSISTENCY_STDEV = 0.10      # ipTM stdev across seeds per candidate
REJECTION_FRACTION = 0.80          # >=80% rejected → veto
DIVERSITY_MIN_PATTERNS = 5         # min distinct anchor patterns for design
TOP_K_FOR_CROSS_CHECK = 5
TOP_K_FOR_SELF_CONSISTENCY = 5

# Anchor amino acids used for diversity pattern fingerprint.
# [critic_config.json verbatim]
ANCHOR_RESIDUES = frozenset(["Y", "W", "F", "H", "R", "K"])

# Iteration cap MUST match graph.CRITIC_ITER_CAP. Critic itself uses the spec
# rule "if new_iters > 2 → halt" (i.e., the third consecutive veto trips halt).
CRITIC_ITER_CAP = 2


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ----------------------------------------------------------------------
# Layer 1: evidence gate
# ----------------------------------------------------------------------

# [L747 verbatim, plus L749 grep confirms branch order at L65,67,73]
def _evidence_gate_artifact(artifact, kind: str, ledger_card_ids: set[str]) -> list[Issue]:
    """Every artifact carrying an evidence_ids list must reference real cards."""
    issues: list[Issue] = []
    ev_ids = getattr(artifact, "evidence_ids", None)
    if ev_ids is None:
        # Not all artifacts carry evidence_ids — skip silently.
        return issues
    if not ev_ids:
        issues.append(Issue(layer="evidence_gate", severity="error",
                            message=f"{kind} has no evidence_ids",
                            suggested_action="Strip artifact or attach evidence."))
        return issues
    missing = [eid for eid in ev_ids if eid not in ledger_card_ids]
    if missing:
        issues.append(Issue(layer="evidence_gate", severity="error",
                            message=f"{kind} references missing evidence_ids: {missing}",
                            suggested_action="Replace with existing cards or commit new ones."))
    return issues


# [Reconstructed from L749 grep: L83 `if unsupported and len(unsupported) > 0.10 * max(1, len(hotspots))`]
def _evidence_gate_hotspots(hotspots: Iterable, ledger_card_ids: set[str]) -> list[Issue]:
    """A hotspot's supported_by_tools list must be non-empty (≥1 source)."""
    issues: list[Issue] = []
    hotspots = list(hotspots) if hotspots else []
    if not hotspots:
        return issues
    unsupported = [h for h in hotspots if not getattr(h, "supported_by_tools", None)]
    if unsupported and len(unsupported) > 0.10 * max(1, len(hotspots)):
        issues.append(Issue(
            layer="evidence_gate", severity="warn",
            message=f"{len(unsupported)} of {len(hotspots)} hotspots have no supported_by_tools",
            suggested_action="Drop unsupported hotspots or attach conservation/SASA evidence.",
        ))
    return issues


# [Reconstructed from L749 grep: L94 `if emap is None`, L98 `if pdb`, L101 `if not cited`]
def _evidence_gate_epitope_map(emap, ledger: list[EvidenceCard], ledger_card_ids: set[str]) -> list[Issue]:
    """The reference PDB used to anchor the epitope map must be cited."""
    issues: list[Issue] = []
    if emap is None:
        return issues
    pdb = getattr(emap, "reference_pdb", None)
    if pdb:
        # Any ledger card whose source_id mentions the PDB ID counts as a citation.
        cited = any(
            (pdb.lower() in (c.source_id or "").lower())
            or (pdb.lower() in (c.source_url or "").lower())
            or (pdb.lower() in (c.claim or "").lower())
            for c in ledger
        )
        if not cited:
            issues.append(Issue(
                layer="evidence_gate", severity="warn",
                message=f"EpitopeMap.reference_pdb={pdb} is not referenced by any EvidenceCard",
                suggested_action=f"Add a PDB EvidenceCard for {pdb} or change reference_pdb.",
            ))
    return issues


# ----------------------------------------------------------------------
# Layer 2: cross-tool agreement
# ----------------------------------------------------------------------

# [L753 verbatim]
def _cross_tool_hotspots(hotspots: list) -> list[Issue]:
    issues: list[Issue] = []
    if not hotspots:
        return issues
    n_anchor = sum(1 for h in hotspots if h.role == "anchor")
    n_low = sum(1 for h in hotspots if h.consensus_score < 2 and h.role in ("anchor", "hub"))
    if n_anchor == 0:
        issues.append(Issue(layer="cross_tool", severity="warn",
                            message="No anchor-class hotspots (consensus_score >= 2)",
                            suggested_action="Lower threshold or run additional conservation analysis."))
    if n_low > 0:
        issues.append(Issue(
            layer="cross_tool", severity="warn",
            message=f"{n_low} anchor/hub residues backed by only 1 tool",
            suggested_action="Mark these as 'rim' until corroborated.",
        ))
    return issues


# [L753 partial + L749 grep L134/L141/L143/L151 + step5 C4 expectation
#  (5 cands, |Δ|=0.27 > 0.15 → veto)]
def _cross_tool_predictions(scores: dict, top_k: int = TOP_K_FOR_CROSS_CHECK) -> list[Issue]:
    """Veto if top-K candidates show systematic Boltz↔Chai-1 disagreement."""
    issues: list[Issue] = []
    if not scores:
        return issues
    ranked = sorted(scores.values(), key=lambda s: -s.composite_score)[:top_k]
    deltas: list[float] = []
    for s in ranked:
        b = s.structural.get("mean_ipTM")
        c = (s.consistency or {}).get("chai_mean_ipTM")
        if b is not None and c is not None:
            deltas.append(abs(float(b) - float(c)))
    if not deltas:
        # Chai-1 cross-check not run yet — surface a hint, not an error.
        issues.append(Issue(
            layer="cross_tool", severity="warn",
            message="Top-K predictions have no Chai-1 cross-check (chai_mean_ipTM)",
            suggested_action="Run Chai-1 on top-5 for cross-tool agreement.",
        ))
        return issues
    mean_delta = mean(deltas)
    if mean_delta > DISAGREEMENT_THRESHOLD:
        issues.append(Issue(
            layer="cross_tool", severity="error",
            message=(f"Top-{len(deltas)} Boltz↔Chai mean |Δ ipTM|={mean_delta:.3f} "
                     f"> {DISAGREEMENT_THRESHOLD}"),
            suggested_action=(
                "Diversity replan: regenerate candidates around different hotspots "
                "or with different generators (BoltzGen / RFdiffusion + ProteinMPNN)."
            ),
        ))
    return issues


# ----------------------------------------------------------------------
# Layer 3: self-consistency
# ----------------------------------------------------------------------

# [L749 grep L173/L181/L183/L185 → predictions-side; step5 implies it produces warns]
def _self_consistency_predictions(predictions: dict, scores: dict) -> list[Issue]:
    """Per-candidate ipTM stdev across diffusion seeds > 0.10 → warn (unstable)."""
    issues: list[Issue] = []
    if not predictions or not scores:
        return issues
    instabilities: list[str] = []
    for cand_id, preds in predictions.items():
        # Each `preds` is a list of ComplexPrediction; pull per-seed iptm values.
        iptms = []
        for p in preds:
            val = getattr(p, "iptm", None) or getattr(p, "ipTM", None)
            if val is not None:
                iptms.append(float(val))
        if len(iptms) >= 2:
            s = pstdev(iptms)
            if s > SELF_CONSISTENCY_STDEV:
                instabilities.append(f"{cand_id} (n={len(iptms)}, σ={s:.3f})")
    if instabilities:
        issues.append(Issue(
            layer="self_consistency", severity="warn",
            message=(f"{len(instabilities)} candidate(s) show ipTM stdev > "
                     f"{SELF_CONSISTENCY_STDEV} across seeds: {', '.join(instabilities[:5])}"
                     + ("…" if len(instabilities) > 5 else "")),
            suggested_action="Use mean ipTM cautiously for these; increase num_samples or downrank.",
        ))
    return issues


def _anchor_pattern(seq: str) -> str:
    """Position-by-position fingerprint: 'A' if residue ∈ ANCHOR_RESIDUES, else '.'.

    Used by _diversity_design. Matches the format dumped at L741:
        ".AAAAAAAA." (10-mer with anchors at positions 2-9).
    """
    return "".join("A" if (aa or "").upper() in ANCHOR_RESIDUES else "." for aa in seq)


# [L749 grep L200/L209; L741 verifies pattern format; step5 C3 (10 cands, 1 anchor pattern → veto)]
def _diversity_design(candidates: list) -> list[Issue]:
    """Veto if a design batch has too few distinct anchor patterns."""
    issues: list[Issue] = []
    if not candidates:
        return issues
    proposed = [c for c in candidates if c.status in ("proposed", "scored")]
    if len(proposed) < 5:
        # Not enough samples to assess diversity meaningfully — silent.
        return issues
    patterns = {_anchor_pattern(c.sequence) for c in proposed}
    if len(patterns) < DIVERSITY_MIN_PATTERNS:
        issues.append(Issue(
            layer="self_consistency", severity="error",
            message=(f"Only {len(patterns)} distinct anchor pattern(s) across "
                     f"{len(proposed)} candidates (min {DIVERSITY_MIN_PATTERNS})"),
            suggested_action=(
                "Design: regenerate with explicit diversity constraints "
                "(mutate ≥3 anchor positions, vary modality, sample new seeds)."
            ),
        ))
    return issues


# ----------------------------------------------------------------------
# Layer 4: calibrated rejection
# ----------------------------------------------------------------------

# [L749 grep L224/L228; step5 C5 (10 cands, 9 rejected → veto)]
def _calibrated_rejection_predictions(scores: dict) -> list[Issue]:
    """Veto if ≥80% of scored candidates land in the 'rejected' confidence class."""
    issues: list[Issue] = []
    if not scores:
        return issues
    n = len(scores)
    n_rejected = sum(1 for s in scores.values() if s.confidence_class == "rejected")
    if n_rejected / n >= REJECTION_FRACTION:
        issues.append(Issue(
            layer="calibrated_rejection", severity="error",
            message=f"{n_rejected}/{n} ({100*n_rejected/n:.0f}%) candidates classified as rejected",
            suggested_action=(
                "Diversity replan: the current design batch fails the confidence floor — "
                "regenerate around different hotspots or with different generators."
            ),
        ))
    return issues


# [L749 grep L241/L245]
def _calibrated_rejection_design(candidates: list) -> list[Issue]:
    """Veto if ≥80% of designed candidates are filtered out before scoring."""
    issues: list[Issue] = []
    if not candidates:
        return issues
    n = len(candidates)
    n_filt = sum(1 for c in candidates if c.status == "filtered_out")
    if n_filt / n >= REJECTION_FRACTION:
        issues.append(Issue(
            layer="calibrated_rejection", severity="error",
            message=f"{n_filt}/{n} ({100*n_filt/n:.0f}%) candidates filtered out before scoring",
            suggested_action="Relax design filters or regenerate with different sampling params.",
        ))
    return issues


# ----------------------------------------------------------------------
# Dispatch + verdict + iteration bookkeeping
# ----------------------------------------------------------------------


def critique(state: AgentState, target_agent: str) -> dict:
    """Run a critic pass *after* the named agent completed.

    target_agent ∈ {research, structure, design, prediction}

    Returns a state-channel dict containing critic_reports/evidence_ledger/
    history append-deltas plus critic_iterations / critic_feedback /
    last_critic_target / halt_reason overrides per the iteration bookkeeping
    rules in critic_config.json.
    """
    ledger = state.get("evidence_ledger", []) or []
    ledger_card_ids = {c.card_id for c in ledger}

    issues: list[Issue] = []
    layers_run: list[str] = []
    target_artifact_id = "n/a"

    # ---- Dispatch [L751 verbatim L274-308] ----
    if target_agent == "research":
        brief = state.get("target_brief")
        if brief is not None:
            target_artifact_id = (brief.uniprot or brief.target_id or "research_brief")
            issues += _evidence_gate_artifact(brief, "TargetBrief", ledger_card_ids)
        layers_run.append("evidence_gate")

    elif target_agent == "structure":
        emap = state.get("epitope_map")
        if emap is not None:
            target_artifact_id = emap.reference_pdb
            issues += _evidence_gate_artifact(emap, "EpitopeMap", ledger_card_ids)
            issues += _evidence_gate_epitope_map(emap, ledger, ledger_card_ids)
        layers_run.append("evidence_gate")
        hotspots = state.get("hotspots", []) or (emap.hotspots if emap else [])
        issues += _evidence_gate_hotspots(hotspots, ledger_card_ids)
        issues += _cross_tool_hotspots(hotspots)
        layers_run.append("cross_tool")

    elif target_agent == "design":
        cands = state.get("candidates", [])
        target_artifact_id = f"{len(cands)}_candidates"
        for c in cands:
            issues += _evidence_gate_artifact(c, f"Candidate {c.cand_id}", ledger_card_ids)
        layers_run.append("evidence_gate")
        issues += _diversity_design(cands)
        layers_run.append("self_consistency")
        issues += _calibrated_rejection_design(cands)
        layers_run.append("calibrated_rejection")

    elif target_agent == "prediction":
        # [Reconstructed: dispatch_order=[evidence_gate, cross_tool, self_consistency, calibrated_rejection]
        #  per critic_config.json; the for-loop visible at L307 is evidence_gate]
        scores = state.get("scores", {})
        preds = state.get("predictions", {})
        target_artifact_id = f"{len(scores)}_scores"
        for s in scores.values():
            issues += _evidence_gate_artifact(s, f"Score {s.cand_id}", ledger_card_ids)
        layers_run.append("evidence_gate")
        issues += _cross_tool_predictions(scores)
        layers_run.append("cross_tool")
        issues += _self_consistency_predictions(preds, scores)
        layers_run.append("self_consistency")
        issues += _calibrated_rejection_predictions(scores)
        layers_run.append("calibrated_rejection")

    # ---- Verdict [L749 grep L319-321] ----
    if any(i.severity == "error" for i in issues):
        verdict = "veto"
    elif any(i.severity == "warn" for i in issues):
        verdict = "warn"
    else:
        verdict = "pass"

    # ---- Recommended action [L749 grep L330-334] ----
    # Prefer the suggestion attached to the most-severe issue.
    rec_action = None
    for sev in ("error", "warn", "info"):
        cand = next(
            (i.suggested_action for i in issues
             if i.severity == sev and i.suggested_action), None)
        if cand:
            rec_action = cand
            break
    if rec_action is None and issues:
        # Fall back to a joined string of all suggestions (v1 behavior).
        rec_action = "; ".join(
            i.suggested_action for i in issues if i.suggested_action
        ) or None

    # ---- Build CriticReport ----
    report = CriticReport(
        target_agent=target_agent,
        target_artifact_id=target_artifact_id,
        layers_run=layers_run,
        issues=issues,
        verdict=verdict,
        recommended_action=rec_action,
    )

    # Drop a card describing the critic decision itself — auditor audits the auditor.
    card = EvidenceCard(
        claim=f"Critic on {target_agent}: {verdict} ({len(issues)} issue(s))",
        source_id=report.report_id,
        source_type="agent_decision",
        tag="SUBJECTIVE",
        confidence=0.9,
        extracted_by="critic",
        payload={"verdict": verdict, "layers": layers_run,
                 "issues": [i.model_dump() for i in issues]},
    )

    # ---- Iteration bookkeeping [L749 grep L364-372; critic_config.json] ----
    out: dict = {
        "critic_reports": [report],
        "evidence_ledger": [card],
        "history": [{"timestamp": _now(), "agent": "critic", "kind": "verdict",
                     "payload": {"target_agent": target_agent, "verdict": verdict}}],
        "last_critic_target": target_agent,
    }
    if verdict == "veto":
        new_iters = int(state.get("critic_iterations", 0)) + 1
        out["critic_iterations"] = new_iters
        out["critic_feedback"] = rec_action
        if new_iters > CRITIC_ITER_CAP:
            out["halt_reason"] = "critic_unconvergent"
    elif verdict == "pass":
        out["critic_iterations"] = 0
        out["critic_feedback"] = None
    # warn: leave critic_iterations/critic_feedback untouched.

    return out
