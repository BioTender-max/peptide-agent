# Architecture overview

PeptideAgent-v0 is a 7-agent system orchestrated by [LangGraph](https://langchain-ai.github.io/langgraph/).
Every cross-agent claim flows through an append-only `EvidenceLedger` so the
Critic can audit and (when needed) veto downstream steps that aren't backed
by ≥2 independent tools.

```
            ┌───────────┐
            │ Supervisor│
            └─────┬─────┘
                  │ routes by state["current_agent"]
   ┌──────────┬───┴───┬──────────┬──────────┬──────────┐
   │          │       │          │          │          │
┌──┴─────┐ ┌──┴────┐┌─┴────┐  ┌──┴───┐  ┌───┴────┐  ┌──┴─────┐
│Planner │ │Research││Struct│  │Design│  │Predict.│  │Reporter│
└────┬───┘ └───┬───┘└──┬───┘  └──┬───┘  └────┬───┘  └────────┘
     │         │       │         │           │
     └─────────┴───────┴─────────┴───────────┘
                       │
                       ▼
                  ┌─────────┐
                  │ Critic  │ ── 4-layer gate ──→ veto / pass
                  └─────────┘
                       │
                       ▼
                Evidence Ledger
            (append-only, hash-dedup)
```

## Agents

| Agent       | Module                         | Responsibility |
|-------------|--------------------------------|----------------|
| Supervisor  | `agents/supervisor.py`         | Reads `state["current_agent"]` and routes to the next node. |
| Planner     | `agents/planner.py`            | Decomposes a target brief into a `TaskPlan` (9 default `TaskNode`s). |
| Research    | `agents/research.py`           | UniProt + RCSB + LiteratureSearch → `TargetBrief` + cards. |
| Structure   | `agents/structure.py`          | Biotite interface analysis + mammalian conservation → `EpitopeMap` + `Hotspot`s. |
| Design      | `agents/design.py`             | Mutation scan + ESM-IF surrogate + cyclic hairpin → `Candidate`s. |
| Prediction  | `agents/prediction.py`         | Boltz Compute API submissions + Chai-1 cross-tool + `ScoreCard`s. |
| Critic      | `agents/critic.py`             | 4-layer evidence/cross-tool/self-consistency/calibrated-reject gate. |
| Reporter    | `agents/reporter.py`           | Renders the final markdown report. |

## State (`peptide_agent/state.py`)

`AgentState` is a `TypedDict` with `Annotated` reducers so parallel branches
merge instead of overwriting:

```python
class AgentState(TypedDict, total=False):
    target_id: str
    target_brief: TargetBrief
    plan: TaskPlan
    hotspots: Annotated[list[Hotspot], op.add]
    epitope_map: EpitopeMap
    candidates: Annotated[list[Candidate], op.add]
    predictions: Annotated[list[ComplexPrediction], op.add]
    scorecards: Annotated[list[ScoreCard], op.add]
    evidence_cards: Annotated[list[EvidenceCard], op.add]
    critic_reports: Annotated[list[CriticReport], op.add]
    iteration: int
    current_agent: str
    halt_reason: str | None
    messages: Annotated[list[BaseMessage], op.add]
```

The `Annotated[list[X], op.add]` reducers keep the agent system safe under
the LangGraph fan-out / fan-in patterns.

## Schemas (`peptide_agent/schemas.py`)

All cross-agent artefacts are Pydantic models, never bare dicts:

- `EvidenceCard`: provenance unit — `card_id`, `claim`, `source_id`,
  `source_type`, `source_url`, `tag` ∈ {VERIFIED, DERIVED, SUBJECTIVE},
  `confidence` ∈ [0, 1], `extracted_by`, `payload`, `content_hash`.
- `TaskPlan` / `TaskNode`: planner output — `plan_id`, `brief`, `nodes`,
  `created_at`, `revisions`.
- `TargetBrief`: `target_id`, `uniprot`, `gene`, `organism`, `length`,
  `sequence`, `function_summary`, `interaction_partners`, `known_binders`,
  `reference_pdbs`, `evidence_ids`.
- `Hotspot`: `chain`, `residue_number`, `residue_aa`, `role` ∈ {anchor, rim,
  hub, ambiguous}, `bsa`, `consensus_score`, `supported_by_tools`,
  `evidence_ids`.
- `EpitopeMap`: `target_id`, `reference_pdb`, `partner_chain`, `hotspots`,
  `summary`, `evidence_ids`.
- `Candidate`: `cand_id`, `sequence`, `length`, `modality`,
  `design_provenance` (`DesignProvenance` block), `intended_hotspots`,
  `evidence_ids`, `status` ∈ {proposed, scored, shortlisted, rejected}.
- `DesignProvenance`: `generator` ∈ {mutation_scan, esm_if, llm_conditional,
  rfdiffusion_proteinmpnn, boltzgen, boltz_protein_design}, `parent_sequence`,
  `parent_pdb`, `parameters`, `seed`, `timestamp`.
- `ComplexPrediction`: per-seed Boltz/Chai-1 output (ipTM, pLDDT_interface,
  pLDDT_peptide, ddG_proxy, predicted_hotspots, evidence_ids).
- `ScoreCard`: `cand_id`, `structural`, `interface`, `energy_proxy`,
  `consistency`, `composite_score`, `confidence_class` ∈ {high, medium, low,
  rejected}, `reasons`, `evidence_ids`.
- `Issue`: `layer`, `severity` ∈ {info, warn, error}, `message`,
  `suggested_action`.
- `CriticReport`: `report_id`, `target_agent`, `target_artifact_id`,
  `layers_run`, `issues`, `verdict` ∈ {pass, soft_warn, veto},
  `recommended_action`, `timestamp`.

## Evidence ledger (`peptide_agent/ledger/store.py`)

```python
class EvidenceLedger:
    def add(self, card: EvidenceCard) -> EvidenceCard: ...
    def by_source(self, source_id: str) -> list[EvidenceCard]: ...
    def by_source_type(self, source_type: str) -> list[EvidenceCard]: ...
    def all(self) -> list[EvidenceCard]: ...
    def to_jsonl(self, path: str) -> None: ...
```

Deduplication is via the SHA-256 `content_hash` field, so an idempotent
rerun of an agent doesn't inflate the trail.

## Graph (`peptide_agent/graph.py`)

```python
def build_graph() -> CompiledStateGraph:
    g = StateGraph(AgentState)
    g.add_node("supervisor", supervisor_node)
    g.add_node("planner", planner_node)
    # ... 6 more agents
    g.add_node("critic", critic_node)
    g.add_node("reporter", reporter_node)

    g.set_entry_point("supervisor")
    g.add_conditional_edges("supervisor", _route_next, _AGENT_TARGETS)
    # Critic edge: if veto → back to planner (one-shot for now)
    g.add_conditional_edges("critic", _critic_route, ...)

    return g.compile()
```

The supervisor routing function consults `state["current_agent"]` and
returns the next node name. The Critic conditional edge is currently a
one-shot loopback: a veto resets to the planner once, but it doesn't yet
loop iteratively (called out under Known caveats).

## Critic 4-layer gate (`peptide_agent/agents/critic.py`)

Each `critique(state, target_agent)` invocation runs the four layers in
order and accumulates `Issue`s. Verdict logic:

| issues                                            | verdict     |
|---------------------------------------------------|-------------|
| 0 errors, 0 warnings                              | `pass`      |
| 0 errors, 1+ warnings                             | `soft_warn` |
| ≥1 error, OR ≥2 warnings                          | `veto`      |

Layers:
1. **evidence_gate** — every claim references at least one `EvidenceCard`.
2. **cross_tool** — high-confidence claims are backed by ≥2 tools;
   `EpitopeMap` references round-trip to the ledger.
3. **self_consistency** — design provenance is consistent across the batch.
4. **calibrated_reject** — applies the configurable rejection thresholds.

Halt cap: if `state["iteration"] > 2` and the verdict is still `veto`, the
critic sets `halt_reason = "critic_unconvergent"` and the graph exits.

## Prediction composite scoring (`peptide_agent/agents/prediction.py`)

Five orthogonal signals are compressed into one composite score:

```python
COMPOSITE_WEIGHTS = {
    "ipTM":                 0.40,
    "hotspot_coverage":     0.20,
    "cross_tool_agreement": 0.15,
    "interface_pLDDT":      0.15,
    "neg_mean_ddG":         0.10,
}
```

If any signal is `None`, its weight drops and the remainder renormalize.
Two override rules eject a candidate to `rejected`:

- `hotspot_coverage < 0.30`
- `|Boltz iPTM − Chai-1 iPTM| > 0.25`

Verified byte-exact against the captured PD-L1 oracle (5 buckets, max
Δ = 1.11 × 10⁻¹⁶).

## Cost discipline (Boltz Compute API)

`tools/boltz_api.py` wraps the Boltz Compute API with two safety rails:

1. `_extract_usd(response)` — robust cost parser that handles numeric,
   string-form, and nested-dict responses (7/7 unit cases).
2. `_maybe_scope_down(per_cand_usd, n_cands, n_seeds)` — if the projected
   batch total exceeds the $30 USD cap, scope down to ≤12 candidates ×
   2 seeds and emit an `EvidenceCard` recording the decision and kept set.

The PD-L1 reference run lands at $2.90 / $30 cap (9.67% utilization).

## Smoke tests

See `tests/test_smoke_all.py`. Section coverage:

- **S1** — 18 module imports
- **S2** — 12 schema instantiations
- **S3** — 6 representative critic cases (C1–C7)
- **S4** — Prediction T1 dry_run (29 → 87 preds) + T2 5-bucket oracle
- **S5** — `_extract_usd` 7 cases
- **S6** — `graph.build_graph()` compiles
- **S7** — `reporter.run(state)` emits non-empty markdown

The legacy `tests/test_smoke.py` 6-test set is kept for the
schemas/ledger/graph subset.
