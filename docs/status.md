# Development status

This file tracks what's actually been built vs. what's still scaffolded, with
honest notes about what was tried, what worked, and what was wrong. It exists
because architecture docs that only describe the target state are misleading
when half the agents haven't actually been exercised against real APIs.

## Step 1 — Scaffold (done)

Implemented under `peptide_agent/`:
- `schemas.py`: Pydantic types — `EvidenceCard`, `TaskPlan`, `TargetBrief`, `EpitopeMap`, `Hotspot`, `Candidate`, `ComplexPrediction`, `ScoreCard`, `Issue`, `CriticReport`, `DesignProvenance`.
- `state.py`: LangGraph `AgentState` TypedDict with `Annotated` reducers (so concurrent updates merge instead of overwriting).
- `ledger/store.py`: append-only `EvidenceLedger` with content-hash deduplication. Two filter APIs: `by_source(source_id)` (exact id match) and `by_source_type(source_type)` (kind match).
- `graph.py`: LangGraph composition. Supervisor node routes between agents based on `state["current_agent"]`.
- 8 agent modules (`planner`, `research`, `structure`, `design`, `prediction`, `critic`, `reporter`, `supervisor`).
- 6 tool wrappers (`uniprot`, `pdb`, `literature`, `interface`, `conservation`, `boltz_api`).

Validation: `tests/test_smoke_all.py` — 7 sections (S1–S7), no network. Confirms
all 18 modules import, 12 Pydantic schemas instantiate, 6 critic cases pass,
prediction T1 dry_run + T2 oracle byte-exact, `_extract_usd` round-trips,
graph compiles, reporter emits a non-empty report. The legacy
`tests/test_smoke.py` (6 tests) is still kept for the schemas/ledger/graph
sub-suite.

## Step 2 — Research + Structure on real APIs (done)

### Research agent

Real external calls:
- UniProt REST: `/uniprotkb/{accession}` for entry data + `/uniprotkb/{accession}.fasta` for canonical sequence.
- RCSB REST: `/rest/v1/core/entry/{pdb}` for metadata, `/files/{pdb}.cif` for structures.
- Literature search via the `LiteratureSearch` tool (Consensus + Exa).

End-to-end on PD-L1: **13 evidence cards in ~3 s**.
- 1 UniProt card (Q9NZQ7, 290 aa, gene CD274)
- 5 PDB cards (4ZQK, 3BIK, 5O45, 5IUS, 5C3T)
- 6 literature cards
- 1 agent_decision card (CA-170 dispute flag)

### Structure agent

Two genuinely independent signals:
- **PDB contact**: Biotite heavy-atom distance, 5 Å cutoff, on the canonical complex.
- **Mammalian conservation**: 8 PD-L1 orthologs (mouse, rat, rabbit, chimp, cow, macaque, pig, dog), pairwise-aligned to human Q9NZQ7 via Biotite `align_optimal` with BLOSUM62, gap=(-10,-1), terminal_penalty=False. Per-position conservation = (match count) / (n_orthologs aligned).

End-to-end on 4ZQK: **22 contact residues, 16 anchors, 6 rim** (~1 s after cache warm).

### What broke and was fixed during step 2

1. **Wrong PDB selected** — initial run used 5C3T as primary PD-L1 reference. 5C3T is a 1-chain PD-L1 IgV monomer (verified via RCSB), not a complex. The interface tool crashed with `ValueError: zero-size array` from `dists.min(axis=1)` on a (N, 0, 3) array. **Fix:** reorder the per-target PDB table to put genuine cocrystals first (`["4ZQK", "3BIK", "5O45", "5IUS", "5C3T"]`) and raise an informative error in `_chain_pair_contacts` when fewer than 2 polymer chains are present.
2. **Hardcoded "literature canonical" hotspot set was fabricated.** The first cut of `structure.py` shipped `literature_hotspots = {54, 56, 66, 113, 115, 121, 122, 123, 124, 125}` attributed to "JACS 2021 / PNAS 2017". Those exact residues come from prior knowledge, not from any retrieved paper in the evidence ledger. **Fix:** delete the set entirely; do not add a third "literature" axis until it can be derived from `EvidenceCard.payload` data.
3. **`confidence` could exceed 1.0.** Original formula `0.7 + 0.1 * consensus` with `consensus = 3` produced 1.0, but with a future 4-axis scoring would breach the Pydantic `≤ 1.0` constraint. **Fix:** clamp via `min(0.99, 0.7 + 0.1 * consensus)`.
4. **`by_source` was ambiguous** — original method filtered by `source_id` but a smoke test expected source-type filtering. **Fix:** add `by_source_type(source_type)` for kind, keep `by_source(source_id)` for exact id.

## Step 3 — Design agent (scaffold + smoke)

Three strategies scaffolded in `agents/design.py`:
- Mutation scan (positional alanine/conservative scan around hotspots)
- ESM-IF surrogate (currently a placeholder grammar; not a real ESM-IF call)
- Cyclic β-hairpin templating (BMS-986189-inspired)

Schemas and grammar pass the smoke suite (S2). Not yet validated against
the rewritten Structure output. Known issue: the ESM-IF surrogate is
grammar-based, not a real inverse-folding model — needs to be swapped for
a genuine inference path or honestly relabelled before paper-style claims.

## Step 4 — Prediction agent (done)

`agents/prediction.py` implements the two-phase Prediction agent:

1. `run(state)` — given a list of `Candidate` objects, build Boltz API
   payloads (target + candidate + intended hotspots) and submit
   asynchronously. Honors a $30 USD batch cap via `_maybe_scope_down`.
2. `collect_and_score(state)` — once Boltz + Chai-1 results return,
   compute interface metrics, ddG proxy, composite score, and `ScoreCard`s.

### Composite scoring

Weighted blend (`COMPOSITE_WEIGHTS`):

| Signal                | Source                          | Weight |
|-----------------------|---------------------------------|--------|
| `ipTM`                | Boltz `iptm_mean`               | 0.40   |
| `hotspot_coverage`    | Predicted ∩ intended hotspots   | 0.20   |
| `cross_tool_agreement`| 1 − \|Boltz iPTM − Chai-1 iPTM\| | 0.15   |
| `interface_pLDDT`     | Mean interface pLDDT / 100      | 0.15   |
| `neg_mean_ddG`        | Clipped (5 − ΔΔG) / 10          | 0.10   |

Missing signals drop their weight and the remainder renormalize. Two
override rules eject a candidate to `rejected`:
- `hotspot_coverage < 0.30`
- `|Boltz iPTM − Chai-1 iPTM| > 0.25`

### Cost scoping (`_maybe_scope_down`)

Decision matrix verified against the live Boltz pricing model
($0.0500/sample at num_samples=1, $0.0334 effective at num_samples=3):

| per_cand | n_cands | n_seeds | full_total | decision                          | kept |
|----------|---------|---------|------------|-----------------------------------|------|
| $0.10    | 29      | 3       | $2.90      | `within_cap`                      | 29   |
| $5.00    | 29      | 3       | $145.00    | `scoped_down_over_cap`            | 12   |
| None     | 29      | 3       | None       | `no_cost_data_proceed_as_planned` | 29   |
| $1.50    | 29      | 3       | $43.50     | `scoped_down_over_cap`            | 12   |
| $0.05    | 29      | 3       | $1.45      | `within_cap`                      | 29   |

The 29 × $0.10 = $2.90 PD-L1 batch is recorded with a 9.67% utilization of
the $30 cap.

### Validation

- **T1 dry_run**: against the captured PD-L1 design output (29 candidates),
  `run()` emits 29 `boltz_start` events and yields 87 `ComplexPrediction`s
  (29 × `num_samples=3`). Evidence cards include the verbatim claim string
  "Prediction Agent submitted 87 Boltz predictions across 29 candidates
  (dry_run=True)."
- **T2 oracle**: composite scores match the captured 5-bucket oracle
  byte-exact (max Δ = 1.11 × 10⁻¹⁶ across high / medium / low /
  reject_coverage / reject_disagreement).
- **T3 cost-scoping matrix**: all 5 cases above pass.
- **T4 ddG clipping**: ΔΔG = +5 and ΔΔG = +10 both produce `ddG_score = 0`;
  ΔΔG = −5 saturates at `1.0`.
- **T5 _classify boundaries**: 14/14 representative classifications agree
  with the spec thresholds.

### Recovery (2026-06-27)

Worker-0 was terminated mid-implementation. `agents/prediction.py` and
`tools/boltz_api.py::_extract_usd` were rebuilt from captured transcript
fragments + on-disk oracle artifacts. Provenance headers mark every
reconstructed function. Three key patches were applied during the
rebuild:

1. `boltz_results[cid]` key fallback chain — handles `iptm_mean`,
   `mean_ipTM`, `mean_ipTM_boltz`, and per-seed lists.
2. `chai1_results[cid]` key fallback — accepts both `chai_ipTM` and `ipTM`.
3. `ScoreCard` evidence payload uses `reasons` (schema field name), not
   `rationale`.

## Step 5 — Critic + Reporter (done)

### Critic (4-layer gate)

`agents/critic.py` implements `critique(state, target_agent)` returning a
`CriticReport`. Four layers run in order; any error or 2+ warnings produce
a `veto`:

1. `evidence_gate` — every claim must reference at least one
   `EvidenceCard` from the ledger.
2. `cross_tool` — high-confidence claims must be backed by ≥2 independent
   tools; `EpitopeMap` references must round-trip through the ledger.
3. `self_consistency` — design provenance must be consistent across the
   batch (no orphan parent_sequences, no contradictory generator
   parameters).
4. `calibrated_reject` — applies the configurable rejection thresholds.

### Validation

The smoke suite covers 6 representative critic cases:
- **C1** research clean → `pass`, 0 iters, layers_run = `['evidence_gate']`.
- **C2** 29-candidate design ran → `veto` (cross_tool tripped).
- **C3** design self_consistency → `veto`, 1 iter.
- **C4** prediction cross_tool → `veto`.
- **C6** halt cap after iter > 2 → `halt_reason = critic_unconvergent`.
- **C7** structure with dangling evidence_id → `veto`, 2 issues,
  recommended_action = `"Replace with existing cards or commit new ones."`
  (verbatim).

### Reporter

`agents/reporter.run(state)` renders a final markdown report from the
state, including:
- Run header (target_id, plan_id, timestamp).
- Per-agent summary section with evidence card counts.
- Critic ledger (verdict, layers_run, issues).
- Shortlisted candidates with their composite scores and confidence
  classes.

Validation: the smoke suite confirms the reporter emits a non-empty
markdown report (2061 B on the minimal smoke state).

## Step 6 — End-to-end PD-L1 run (pending)

The single remaining gap. Each agent is now individually validated; the
end-to-end PD-L1 run that exercises Planner → Research → Structure →
Design → Prediction → Critic → Reporter against real APIs has not been
re-run after the v2 recovery. Expected next step.

## Open caveats (lifted to README)

See `README.md § Known caveats` for the user-facing list.
