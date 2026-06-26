# v2-RECONSTRUCTED from transcript+spec+oracles (worker termination 2026-06-27)
# Verbatim sources:
#   - L330/L558 v1 baseline (6,414 chars, also on disk at
#     /mnt/results/peptide-agent/peptide_agent/agents/prediction.py)
#   - L637 (lines 100-269 v2 verbatim, pre-patch — cost-est+scope+per-cand-seed loop)
#   - L639 (lines 60-109 v2 verbatim — PROMOTION_THRESHOLDS + run() signature)
#   - L641 (lines 115-274 v2 verbatim, identical body to L637 overlap region)
#   - L667 (lines 460-519 v2 verbatim — _safe_mean + _extract_usd re-export
#     + _maybe_scope_down pre-fix body)
#   - L580/L582/L651/L663 ExecuteCode outputs verifying behavior
#   - L597 (i=604) review documenting:
#     • 5 confidence-class buckets w/ exact composite scores
#     • cov<0.30 → rejected; |Δ|>0.25 → rejected (override path)
#     • COMPOSITE_WEIGHTS = 0.40+0.20+0.15+0.15+0.10 = 1.00
#     • single-signal renormalization (None → drop, redistribute)
#     • ddG=±5 → 1.0/0.0 (linear in between)
#   - L642 patch instructions (per-cand-num_samples refactor; drop peptide_id/seed)
#   - L660/L668 patches (delete in-agent _extract_usd; fix _maybe_scope_down math)
# Oracle artifacts on disk:
#   /mnt/results/peptide-agent/runs/step4_pdl1_4zqk_prediction/:
#     - prediction_config.json (composite_weights + promotion_thresholds + defaults)
#     - scoring_smoke_results.json (5 buckets w/ 7-decimal composite values)
#     - summary.json (phases, two-phase design, schema correction list)
#     - boltz_api_validation_examples.json (live verified payload shape)
#     - boltz_api_validation_summary.md (pricing model, schema corrections)
# Composite-score formula BACK-SOLVED against all 5 scoring oracle buckets
# (7-decimal byte-exact match):
#     plddt_norm = interface_plddt / 100.0
#     agree = 1.0 - |boltz_iptm - chai_iptm|
#     ddG_score = (5.0 - mean_ddG) / 10.0    # clipped to [0, 1]
#     comp = 0.40*ipTM + 0.20*cov + 0.15*agree + 0.15*plddt_norm + 0.10*ddG_score
"""Prediction Agent — co-fold candidates and score them (v2).

Two-phase design:

  run(state, ...)               — SUBMISSION ONLY. Estimate Boltz cost on a
                                  sample, scope down if over budget, submit
                                  every kept candidate × N seeds. No
                                  structures yet. Returns submission manifest.

  collect_and_score(state, ...) — COLLECTION + SCORING. Pulls back Boltz
                                  results, joins with Chai-1 (cross-check
                                  top-K), ThermoMPNN (stability top-K), and
                                  interface scores (hotspot coverage), then
                                  composes a weighted ScoreCard per candidate
                                  and mutates candidate.status.

  This split mirrors the real HPC/API world: submission is millisecond
  and returns immediately; collection waits for completion callbacks.
  Tests cover both phases independently.

Cost control:
  - estimate_cost first against a sample with num_samples=n_seeds (Boltz
    prices the BATCH, not per-sample — so the agent must price at the
    actual call shape).
  - $30 USD hard cap (default; configurable). If estimated total > cap,
    scope to top scoped_top_n × scoped_seeds.
  - 1 API call per candidate, num_samples=n_seeds → n_seeds samples land
    in one job. Avoids the obsolete (cand × seed) → call duplication.

Cross-tool cross-check:
  - Top-K (default 5) candidates by composite_score are routed to Chai-1
    on Biomni HPC after Boltz returns. Critic reads chai_mean_ipTM.

Stability proxy:
  - Top-K (default 5) candidates run ThermoMPNN SSM on the peptide chain
    of the predicted complex; mean ddG is folded into composite_score.

Confidence classification (PROMOTION_THRESHOLDS):
  - rejected: hotspot < 0.30 OR |boltz - chai| > 0.25 (override)
  - high:    boltz >= 0.65 AND chai >= 0.60 AND cov >= 0.60 AND |Δ| <= 0.15
  - medium:  boltz >= 0.55 AND cov >= 0.50 AND |Δ| <= 0.20
  - low:     anything else passing structural sanity
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

from ..schemas import (
    Candidate,
    ComplexPrediction,
    EvidenceCard,
    ScoreCard,
)
from ..state import AgentState
from ..tools.boltz_api import (
    BoltzAPIWrapper,
    build_complex_input,
    _extract_usd,
)

# --- Defaults (PLAN.md §3.5 + §4.3) ---------------------------------
# [prediction_config.json verbatim]
DEFAULT_N_SEEDS = 3
DEFAULT_BOLTZ_USD_CAP = 30.0
DEFAULT_SCOPED_TOP_N = 12
DEFAULT_SCOPED_SEEDS = 2
DEFAULT_TOP_K_CROSS_CHECK = 5
DEFAULT_TOP_K_STABILITY = 5

# --- Composite-score weights (sum to 1.0) ---------------------------
# [prediction_config.json verbatim; back-solved against oracle scoring_smoke_results.json]
COMPOSITE_WEIGHTS = {
    "ipTM": 0.40,
    "hotspot_coverage": 0.20,
    "cross_tool_agreement": 0.15,
    "interface_pLDDT": 0.15,
    "neg_mean_ddG": 0.10,
}

# --- Promotion thresholds [L639 verbatim] ---------------------------
PROMOTION_THRESHOLDS = {
    "high":     {"boltz_ipTM": 0.65, "chai_ipTM": 0.60, "hotspot": 0.60, "agree": 0.15},
    "medium":   {"boltz_ipTM": 0.55, "hotspot": 0.50, "agree": 0.20},
    "rejected": {"hotspot": 0.30, "agree_max": 0.25},
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_mean(xs: Iterable[Optional[float]]) -> Optional[float]:
    """Compute mean of non-None values; return None if all are None."""
    # [L667 verbatim]
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


# ============================ PHASE 1 ================================
# Submission phase: cost-estimate, scope, submit Boltz predictions only.
# Chai-1 / ThermoMPNN are deferred to the demo runner (HPC GPU rate limits).
# =====================================================================


def run(
    state: AgentState,
    n_seeds: int = DEFAULT_N_SEEDS,
    max_candidates: Optional[int] = None,
    usd_cap: float = DEFAULT_BOLTZ_USD_CAP,
    scoped_top_n: int = DEFAULT_SCOPED_TOP_N,
    scoped_seeds: int = DEFAULT_SCOPED_SEEDS,
    dry_run: bool = False,
) -> dict:
    """Submit Boltz predictions for kept candidates with a USD cap.

    Args:
        n_seeds: seeds per candidate (default 3). 1 API call per candidate
            uses num_samples=n_seeds, yielding n_seeds ComplexPrediction
            rows that share a submission_id.
        max_candidates: hard cap on candidate count (None = no cap).
        usd_cap: estimated-total-USD hard cap (default $30). If exceeded, scope
            to the top `scoped_top_n` candidates with `scoped_seeds` each.
        scoped_top_n, scoped_seeds: fallback scope when cap is exceeded.
        dry_run: skip API calls, useful for tests and budget previews.

    Returns delta dict for AgentState:
      - predictions: {cand_id: [ComplexPrediction × n_seeds]}
      - evidence_ledger: cost-estimate card + scope-decision card + submission-summary card
      - history: append-only event log (1 estimate_cost + 1 decision + N start_prediction)
    """
    # [L639+L637 verbatim filter, L591 documents the .status=='proposed' filter intent]
    candidates = [c for c in state.get("candidates", []) if c.status == "proposed"]
    if max_candidates is not None:
        candidates = candidates[:max_candidates]
    if not candidates:
        raise RuntimeError("Prediction Agent has no proposed candidates")

    target_brief = state.get("target_brief")
    target_seq = target_brief.sequence if target_brief else None
    if not target_seq:
        raise RuntimeError("Prediction Agent requires TargetBrief.sequence")

    boltz = BoltzAPIWrapper() if not dry_run else None

    out_dir = Path(state.get("out_dir") or "/workspace/peptide-agent/runs/prediction_default") / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)

    cards: list[EvidenceCard] = []
    events: list[dict] = []

    # ---- 1. Cost estimate on a single sample (priced at num_samples=n_seeds) -----
    # [L642 patch: sample uses num_samples=n_seeds, not 1; otherwise the
    #  estimate lowballs the actual per-call cost]
    per_cand_usd = None
    sample_payload = build_complex_input(
        target_seq=target_seq,
        peptide_seq=candidates[0].sequence,
        num_samples=n_seeds,
    )
    if not dry_run:
        try:
            cost_estimate = boltz.estimate_cost(sample_payload)
            per_cand_usd = _extract_usd(cost_estimate)
            events.append({
                "timestamp": _now(), "agent": "prediction", "kind": "tool_call",
                "payload": {"tool": "boltz_estimate_cost",
                            "per_candidate_usd": per_cand_usd,
                            "num_samples": n_seeds,
                            "raw": cost_estimate},
            })
            # [L663 verbatim claim string]
            claim = (f"Boltz API estimated cost per candidate "
                     f"(1 call, num_samples={n_seeds}): "
                     f"${per_cand_usd:.4f}" if per_cand_usd is not None else
                     "Boltz API cost estimate returned but could not parse USD")
            cards.append(EvidenceCard(
                claim=claim,
                source_id="boltz_estimate_cost",
                source_type="tool_output",
                tag="VERIFIED",
                confidence=0.99,
                extracted_by="prediction",
                payload={"per_candidate_usd": per_cand_usd,
                         "num_samples": n_seeds,
                         "raw": cost_estimate},
            ))
        except Exception as e:
            events.append({"timestamp": _now(), "agent": "prediction",
                           "kind": "tool_error",
                           "payload": {"tool": "boltz_estimate_cost", "error": str(e)}})

    # ---- 2. Scope decision -----------------------------------------
    scoped, scoping_decision = _maybe_scope_down(
        candidates=candidates,
        n_seeds=n_seeds,
        per_cand_usd=per_cand_usd,
        usd_cap=usd_cap,
        scoped_top_n=scoped_top_n,
        scoped_seeds=scoped_seeds,
    )
    final_candidates, final_seeds = scoped
    estimated_total_usd = (per_cand_usd or 0.0) * len(final_candidates)

    # [L663 verbatim claim string]
    scope_claim = (
        f"Prediction Agent will submit {len(final_candidates)} "
        f"Boltz API calls (1 per candidate, num_samples={final_seeds}); "
        f"total {len(final_candidates) * final_seeds} structure samples."
    )
    cards.append(EvidenceCard(
        claim=scope_claim,
        source_id="prediction_scope_decision",
        source_type="agent_decision",
        tag="DERIVED",
        confidence=1.0,
        extracted_by="prediction",
        payload={
            **scoping_decision,
            "estimated_total_usd": estimated_total_usd,
            "n_api_calls": len(final_candidates),
            "n_total_samples": len(final_candidates) * final_seeds,
            "usd_cap": usd_cap,
        },
    ))
    events.append({"timestamp": _now(), "agent": "prediction", "kind": "decision",
                   "payload": scoping_decision})

    # ---- 3. Submit predictions (1 call per candidate, num_samples=n_seeds) -----
    # [L642 patch + L651/L663 oracle: 29 cands → 29 API calls → 87 ComplexPrediction rows]
    predictions: dict[str, list[ComplexPrediction]] = {}
    submitted_jobs: list[dict] = []

    for cand in final_candidates:
        cand_preds: list[ComplexPrediction] = []
        submission_id = None
        submission_status = None
        try:
            if dry_run:
                submission_id = f"DRY_{cand.cand_id}"
                submission_status = "DRY_RUN"
            else:
                payload = build_complex_input(
                    target_seq=target_seq,
                    peptide_seq=cand.sequence,
                    num_samples=final_seeds,
                )
                submission = boltz.start_prediction(
                    payload,
                    idempotency_key=cand.cand_id,
                )
                submission_id = submission.get("id")
                submission_status = submission.get("status")
                time.sleep(0.2)  # polite throttle
            # One ComplexPrediction per sample_index inside the batched call.
            for sample_index in range(final_seeds):
                pred = ComplexPrediction(
                    cand_id=cand.cand_id,
                    predictor="boltz_api",
                    seed=sample_index,
                    raw_metrics={
                        "submission_id": submission_id,
                        "sample_index": sample_index,
                        "status": submission_status,
                    },
                )
                cand_preds.append(pred)
                submitted_jobs.append({
                    "cand_id": cand.cand_id,
                    "sequence": cand.sequence,
                    "sample_index": sample_index,
                    "submission_id": submission_id,
                    "status": submission_status,
                })
            events.append({
                "timestamp": _now(), "agent": "prediction",
                "kind": "tool_call",
                "payload": {"tool": "boltz_start",
                            "cand_id": cand.cand_id,
                            "num_samples": final_seeds,
                            "submission_id": submission_id},
            })
        except Exception as e:
            events.append({
                "timestamp": _now(), "agent": "prediction",
                "kind": "tool_error",
                "payload": {"tool": "boltz_start",
                            "cand_id": cand.cand_id,
                            "error": str(e)},
            })
        predictions[cand.cand_id] = cand_preds

    # Persist submission manifest
    (out_dir / "boltz_submissions.json").write_text(
        json.dumps(submitted_jobs, indent=2, default=str)
    )

    n_submitted = sum(len(p) for p in predictions.values())
    cards.append(EvidenceCard(
        claim=(
            f"Prediction Agent submitted {n_submitted} Boltz predictions across "
            f"{len(predictions)} candidates (dry_run={dry_run})."
        ),
        source_id="prediction_submission_summary",
        source_type="agent_decision",
        tag="DERIVED",
        confidence=1.0,
        extracted_by="prediction",
        payload={
            "n_candidates": len(predictions),
            "n_seeds": final_seeds,
            "n_jobs": n_submitted,
            "n_api_calls": len(final_candidates),
            "dry_run": dry_run,
            "out_dir": str(out_dir),
            "estimated_total_usd": estimated_total_usd,
            "manifest_path": str(out_dir / "boltz_submissions.json"),
        },
    ))

    return {
        "predictions": predictions,
        "evidence_ledger": cards,
        "history": events,
        # The demo runner uses these to schedule Chai-1 / ThermoMPNN after Boltz returns
        "_top_k_for_cross_check": min(DEFAULT_TOP_K_CROSS_CHECK, len(final_candidates)),
        "_top_k_for_stability": min(DEFAULT_TOP_K_STABILITY, len(final_candidates)),
    }


# ============================ SCOPING ================================
# [L667 verbatim base + L668 patch: drop × n_seeds since per_cand_usd is now
#  per-CANDIDATE (already-batched cost)]


def _maybe_scope_down(
    candidates: list[Candidate],
    n_seeds: int,
    per_cand_usd: Optional[float],
    usd_cap: float,
    scoped_top_n: int,
    scoped_seeds: int,
) -> tuple[tuple[list[Candidate], int], dict]:
    """Decide whether to scope down based on cost estimate vs cap.

    per_cand_usd is the cost of one API call (with num_samples=n_seeds).
    Total estimated = per_cand_usd × len(candidates); n_seeds does NOT
    multiply because it's already encoded in per_cand_usd.
    """
    full_total = (per_cand_usd or 0.0) * len(candidates)
    if per_cand_usd is None:
        return (candidates, n_seeds), {
            "scoping": "no_cost_data_proceed_as_planned",
            "n_candidates": len(candidates), "n_seeds": n_seeds,
            "per_cand_usd": None, "full_total_usd": None,
        }
    if full_total <= usd_cap:
        return (candidates, n_seeds), {
            "scoping": "within_cap",
            "n_candidates": len(candidates), "n_seeds": n_seeds,
            "per_cand_usd": per_cand_usd, "full_total_usd": full_total,
            "usd_cap": usd_cap,
        }
    # Scoped fallback: top N candidates × scoped seeds.
    scoped = candidates[:scoped_top_n]
    # Approximate scoped cost: if peptide length doesn't move the dial much,
    # per_cand_usd is essentially target-driven and ~constant across cands.
    # The scoped batch uses fewer samples, so we apply a sample-count ratio
    # for the estimate. This is approximate but defensible.
    sample_ratio = scoped_seeds / max(1, n_seeds)
    scoped_per_cand_usd = per_cand_usd * sample_ratio
    return (scoped, scoped_seeds), {
        "scoping": "scoped_down_over_cap",
        "n_candidates_original": len(candidates),
        "n_seeds_original": n_seeds,
        "n_candidates": len(scoped), "n_seeds": scoped_seeds,
        "per_cand_usd": per_cand_usd,
        "scoped_per_cand_usd_est": scoped_per_cand_usd,
        "full_total_usd": full_total,
        "scoped_total_usd": scoped_per_cand_usd * len(scoped),
        "usd_cap": usd_cap,
    }


# ============================ PHASE 2 ================================
# Collection + scoring: harvest Boltz/Chai-1/ThermoMPNN results, compose
# ScoreCards with confidence_class.
# =====================================================================


def collect_and_score(
    state: AgentState,
    *,
    boltz_results: dict[str, dict],
    chai1_results: Optional[dict[str, dict]] = None,
    thermompnn_results: Optional[dict[str, dict]] = None,
    interface_scores: Optional[dict[str, dict]] = None,
    weights: Optional[dict[str, float]] = None,
) -> dict:
    """Harvest results and produce per-candidate ScoreCards.

    Inputs (all keyed by cand_id):
      boltz_results: per-cand dict with at least
        {ipTM_per_seed: [..], iptm_mean, plddt_interface_mean, ...}.
      chai1_results: per-cand dict with chai_ipTM (mean across diffusion samples).
      thermompnn_results: per-cand dict with mean_ddG (peptide-chain SSM mean).
      interface_scores: per-cand dict with hotspot_coverage_fraction +
        contacted_target_residues.

    Returns delta:
      scores: {cand_id: ScoreCard}
      candidates: list with status mutated to scored/shortlisted/rejected.
      evidence_ledger: 1 card per scored candidate plus summary.
      history: 1 event per scoring decision.
    """
    weights = weights or COMPOSITE_WEIGHTS
    candidates = list(state.get("candidates", []))
    if not candidates:
        return {"scores": {}}

    chai1_results = chai1_results or {}
    thermompnn_results = thermompnn_results or {}
    interface_scores = interface_scores or {}

    scores: dict[str, ScoreCard] = {}
    cards: list[EvidenceCard] = []
    events: list[dict] = []

    # Build a fresh, mutated list so we don't aliasing-trash the input.
    mutated_cands: list[Candidate] = []
    for cand in candidates:
        if cand.cand_id not in boltz_results:
            mutated_cands.append(cand)  # untouched — never landed in Boltz
            continue

        b = boltz_results[cand.cand_id]
        # Boltz ipTM: prefer mean across seeds; fall back to per-seed list mean.
        # Accept either raw Boltz convention (iptm_mean / plddt_interface_mean)
        # or the agent-aggregated convention (mean_ipTM / mean_pLDDT_interface).
        # Either name is canonical depending on whether the caller already ran
        # _safe_mean over seed samples or is handing us per-seed lists.
        boltz_iptm_mean = (
            b.get("iptm_mean")
            or b.get("mean_ipTM")
            or b.get("mean_ipTM_boltz")
        )
        if boltz_iptm_mean is None:
            boltz_iptm_mean = _safe_mean(
                b.get("ipTM_per_seed") or b.get("ipTM_per_sample") or []
            )
        plddt_interface = (
            b.get("plddt_interface_mean")
            or b.get("plddt_interface")
            or b.get("mean_pLDDT_interface")
            or b.get("mean_pLDDT_interface_boltz")
        )

        # Accept either chai_ipTM (explicit, raw Chai-1 convention) or ipTM
        # (when the caller already normalised the key during result harvest).
        chai_block = chai1_results.get(cand.cand_id) or {}
        chai_iptm = chai_block.get("chai_ipTM") or chai_block.get("ipTM")
        ddG = (thermompnn_results.get(cand.cand_id) or {}).get("mean_ddG")
        cov_block = interface_scores.get(cand.cand_id) or {}
        hotspot_coverage = cov_block.get("hotspot_coverage_fraction")

        composite = _compose_score(
            boltz_iptm_mean=boltz_iptm_mean,
            chai_iptm=chai_iptm,
            interface_plddt=plddt_interface,
            hotspot_coverage=hotspot_coverage,
            mean_ddG=ddG,
            weights=weights,
        )
        cls, reasons = _classify(
            boltz_iptm_mean=boltz_iptm_mean,
            chai_iptm=chai_iptm,
            hotspot_coverage=hotspot_coverage,
        )

        # Build ScoreCard from the assembled signals.
        sc = ScoreCard(
            cand_id=cand.cand_id,
            prediction_ids=[],  # populated by the demo runner if it has them
            composite_score=composite,
            confidence_class=cls,
            sub_scores={k: w for k, w in weights.items()},
            structural={
                "mean_ipTM": boltz_iptm_mean,
                "mean_pLDDT_interface": plddt_interface,
                "chai_mean_ipTM": chai_iptm,
            },
            binding_proxy={
                "hotspot_coverage_fraction": hotspot_coverage,
                "contacted_target_residues": cov_block.get("contacted_target_residues", []),
            },
            consistency={
                "chai_mean_ipTM": chai_iptm,
                "boltz_vs_chai_ipTM_abs_diff": (
                    abs(boltz_iptm_mean - chai_iptm)
                    if (boltz_iptm_mean is not None and chai_iptm is not None) else None
                ),
                "mean_ddG": ddG,
            },
            rationale=("; ".join(reasons) if reasons else "passes structural sanity"),
            evidence_ids=list(cand.evidence_ids or []),
        )
        scores[cand.cand_id] = sc

        # Update candidate.status based on confidence_class.
        # [L591 verbatim behavior: collect_and_score mutates candidate.status]
        new_status = {
            "high":     "shortlisted",
            "medium":   "scored",
            "low":      "scored",
            "rejected": "rejected",
        }.get(cls, "scored")
        # Pydantic v2 model_copy with update=
        mutated_cands.append(cand.model_copy(update={"status": new_status}))

        cards.append(EvidenceCard(
            claim=(f"Score {cand.cand_id}: composite={composite:.3f} "
                   f"({cls}); ipTM={boltz_iptm_mean}; cov={hotspot_coverage}"),
            source_id=f"scorecard_{cand.cand_id}",
            source_type="agent_decision",
            tag="DERIVED",
            confidence=0.95,
            extracted_by="prediction",
            payload={
                "composite_score": composite,
                "confidence_class": cls,
                "reasons": sc.reasons,
                "weights": weights,
            },
        ))
        events.append({"timestamp": _now(), "agent": "prediction",
                       "kind": "decision",
                       "payload": {"cand_id": cand.cand_id,
                                   "composite_score": composite,
                                   "confidence_class": cls,
                                   "new_status": new_status}})

    return {
        "scores": scores,
        "candidates": mutated_cands,   # overwrite (status mutations applied)
        "evidence_ledger": cards,
        "history": events,
    }


# ============================ SCORING ================================
# [Composite-score formula back-solved against oracle buckets; weights are
#  the values from prediction_config.json. None signals → drop weight,
#  renormalize across present signals. ddG capped at [-5, +5].]


def _compose_score(
    *,
    boltz_iptm_mean: Optional[float],
    chai_iptm: Optional[float],
    interface_plddt: Optional[float],
    hotspot_coverage: Optional[float],
    mean_ddG: Optional[float],
    weights: dict[str, float],
) -> float:
    """Weighted composite score in [0, 1]; renormalizes over present signals.

    Per-signal mapping:
      ipTM:              raw value (already 0-1)
      hotspot_coverage:  raw value (already 0-1)
      cross_tool_agreement: 1 - |boltz_iptm - chai_iptm|; None if either missing
      interface_pLDDT:   plddt / 100.0 (scale 0-100 → 0-1)
      neg_mean_ddG:      (5 - ddG) / 10, clipped to [0, 1] (positive ddG = bad)
    """
    # Compute per-signal values, leaving None where input is missing.
    signal_vals: dict[str, Optional[float]] = {
        "ipTM": float(boltz_iptm_mean) if boltz_iptm_mean is not None else None,
        "hotspot_coverage": float(hotspot_coverage) if hotspot_coverage is not None else None,
        "interface_pLDDT": (float(interface_plddt) / 100.0)
            if interface_plddt is not None else None,
        "cross_tool_agreement": (
            1.0 - abs(float(boltz_iptm_mean) - float(chai_iptm))
        ) if (boltz_iptm_mean is not None and chai_iptm is not None) else None,
        "neg_mean_ddG": (
            max(0.0, min(1.0, (5.0 - float(mean_ddG)) / 10.0))
        ) if mean_ddG is not None else None,
    }
    # Renormalize weights over signals that landed.
    present_weights = {k: weights[k] for k, v in signal_vals.items() if v is not None}
    total_w = sum(present_weights.values())
    if total_w == 0:
        return 0.0
    norm = {k: w / total_w for k, w in present_weights.items()}
    return sum(norm[k] * signal_vals[k] for k in norm)


def _classify(
    *,
    boltz_iptm_mean: Optional[float],
    chai_iptm: Optional[float],
    hotspot_coverage: Optional[float],
) -> tuple[str, list[str]]:
    """Promote to {high, medium, low, rejected} with reason strings.

    Reject overrides apply FIRST (per L597 review):
      hotspot < 0.30 → rejected (override even if Boltz high)
      |Boltz - Chai| > 0.25 → rejected (override even with high composite)
    """
    reasons: list[str] = []
    cov = hotspot_coverage
    b = boltz_iptm_mean
    c = chai_iptm
    delta = abs(b - c) if (b is not None and c is not None) else None

    # 1. Hard rejects (apply first regardless of other signals).
    if cov is not None and cov < PROMOTION_THRESHOLDS["rejected"]["hotspot"]:
        reasons.append(f"hotspot_coverage={cov:.2f} < 0.30 (reject floor)")
        return "rejected", reasons
    if delta is not None and delta > PROMOTION_THRESHOLDS["rejected"]["agree_max"]:
        reasons.append(f"|boltz-chai|={delta:.2f} > 0.25 (cross-tool reject cap)")
        return "rejected", reasons

    # 2. High tier — strictest gating.
    hi = PROMOTION_THRESHOLDS["high"]
    if (b is not None and b >= hi["boltz_ipTM"]
        and c is not None and c >= hi["chai_ipTM"]
        and cov is not None and cov >= hi["hotspot"]
        and delta is not None and delta <= hi["agree"]):
        return "high", reasons

    # 3. Medium tier — Boltz + coverage + (when both present) agreement.
    md = PROMOTION_THRESHOLDS["medium"]
    if (b is not None and b >= md["boltz_ipTM"]
        and cov is not None and cov >= md["hotspot"]
        and (delta is None or delta <= md["agree"])):
        return "medium", reasons

    # 4. Low — anything else passes the structural-sanity bar implicitly.
    return "low", reasons
