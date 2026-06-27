# Changelog

All notable changes to PeptideAgent-v0 are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `tests/test_smoke_all.py` — comprehensive 7-section smoke suite covering
  module imports (18), schema instantiation (12), Critic cases (6), Prediction
  T1 + T2 (29 → 87 + 5-bucket oracle byte-exact), `_extract_usd` (7),
  graph compile, and reporter emission.
- `_extract_usd` helper in `tools/boltz_api.py` — robustly parses Boltz cost
  responses including string-form numerics (`"0.0500"`) and nested
  `{cost: {usd: …}}` shapes. 7/7 unit cases pass.
- Composite-score documentation in README with the five weighted signals
  and the two override rules.

### Changed
- `agents/prediction.py` reconstructed to the v2 spec with the two-phase
  `run()` + `collect_and_score()` API. Verified against the captured
  PD-L1 oracle at byte-exact precision (max Δ = 1.11 × 10⁻¹⁶ across the
  5 high / medium / low / reject_coverage / reject_disagreement buckets).
  Status mutation confirmed: `high → shortlisted`, `medium → scored`,
  `low → scored`, override-reject → `rejected`.
- `agents/critic.py` reconstructed to the v2 spec with the 4-layer gate
  (`evidence_gate`, `cross_tool`, `self_consistency`, `calibrated_reject`).
  All 8 representative cases (C1–C7) pass, including the verbatim
  recommended_action string
  `"Replace with existing cards or commit new ones."` for dangling
  evidence_ids.
- Prediction cost-scoping decision now exposes one of
  `within_cap`, `scoped_down_over_cap`, `no_cost_data_proceed_as_planned`
  via an `EvidenceCard`, with kept candidates and seeds recorded for audit.

### Fixed
- `boltz_results` key fallback chain: accept `iptm_mean`, `mean_ipTM`, and
  `mean_ipTM_boltz`; also derive from `ipTM_per_seed` / `ipTM_per_sample`
  when only per-seed lists are present.
- `chai1_results` key fallback: accept both `chai_ipTM` and `ipTM` so the
  cross-tool agreement signal survives upstream renames.
- `ScoreCard` evidence payload uses the schema field `reasons` (previous
  code referenced a non-existent `rationale` attribute).
- `_extract_usd` now correctly parses `estimated_cost_usd: "0.0500"` shape
  observed from the live Boltz API (was returning `None` before).

### Recovery notes
- Worker-0 was terminated mid-implementation on 2026-06-27. The Prediction
  agent + critic plus the `_extract_usd` boltz_api helper were rebuilt from
  the captured transcript fragments + on-disk oracle artifacts. Every
  reconstructed function carries an inline provenance header.
- The provenance trail is documented step-by-step in
  `docs/status.md § Step 4 — Prediction agent`.

## [0.1.0a1] — 2026-06-27

Initial alpha — pre-recovery scaffold. Captures the early v2 state with:
- Research + Structure agents validated on PD-L1 (4ZQK) against real APIs.
- Design / Prediction / Critic / Reporter agents scaffolded but unvalidated.
- 6/6 `tests/test_smoke.py` smoke tests passing (no network).
- MIT-licensed, OSI-compliant, BioTender-aligned.
