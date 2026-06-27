# PeptideAgent-v0

> Agentic peptide-binder discovery with a tracked evidence trail.

PeptideAgent is a [LangGraph](https://langchain-ai.github.io/langgraph/)-based
7-agent system that takes a protein target + epitope brief and runs deep
literature/structure research → epitope mapping → multi-strategy peptide
design → in-silico evaluation → self-critique → final report.

The architectural focus is **provenance**: every claim a downstream agent uses
is backed by an `EvidenceCard` with a `source_id`, `source_url`, `tag`
(VERIFIED / DERIVED / SUBJECTIVE), and a `confidence` score. The Critic is
allowed to veto anything not backed by ≥2 independent tools.

## Status

This is alpha-quality research code. Numbers below are not steady-state benchmarks; they reflect the demo run on the project's reference case (PD-L1, PDB 4ZQK).

| Subsystem | State | Validation |
|---|---|---|
| Scaffold (schemas, ledger, graph, supervisor) | Done | 12/12 schemas instantiate; graph compiles |
| Research agent (UniProt + RCSB + literature search) | Real APIs | End-to-end on PD-L1: 13 cards in ~3 s |
| Structure agent (Biotite interface + mammalian conservation) | Real APIs | End-to-end on 4ZQK: 22 contact residues, 2-axis consensus |
| Design agent (mutation scan + ESM-IF surrogate + cyclic hairpin) | Scaffold + smoke | Schemas + grammar verified; ESM-IF still placeholder |
| Prediction agent (Boltz Compute API wrapper + composite scoring) | Done | T1 dry_run (29 → 87 preds), T2 oracle byte-exact (Δ ≤ 1.11e-16, 5/5 buckets), T3/T4/T5 scope-down + edge cases |
| Critic (4-layer gate) | Done | 8/8 critic cases (C1–C7) incl. cross-tool veto with verbatim recommended_action |
| Reporter (Markdown render) | Done | Emits non-empty report from a minimal state |

See [`docs/status.md`](docs/status.md) for the full step-by-step development
log and known caveats, and [`CHANGELOG.md`](CHANGELOG.md) for the post-v2
recovery summary.

## Layout

```
peptide_agent/
  __init__.py            # public API: build_graph, EvidenceLedger
  __main__.py            # CLI entry point: python -m peptide_agent
  schemas.py             # Pydantic types: EvidenceCard, TaskPlan, EpitopeMap, …
  state.py               # LangGraph AgentState TypedDict (Annotated reducers)
  graph.py               # LangGraph composition with conditional edges
  ledger/store.py        # Append-only EvidenceLedger w/ content-hash dedup
  agents/                # 7 agents + supervisor router
    planner.py           research.py    structure.py    design.py
    prediction.py        critic.py      reporter.py     supervisor.py
  tools/                 # Tool wrappers — every external call goes through here
    uniprot.py           pdb.py         literature.py   interface.py
    conservation.py      boltz_api.py
tests/
  test_smoke.py          # Original 6-test smoke set (no network)
  test_smoke_all.py      # 7-section comprehensive suite (S1–S7)
runs/                    # Per-run outputs (gitignored)
cache/                   # Tool caches, e.g. conservation_<accession>.json
docs/                    # Architecture + development log
```

## Install

```bash
git clone https://github.com/<your-org>/peptide-agent.git
cd peptide-agent

# Recommended: a fresh virtualenv on Python ≥ 3.10
python -m venv .venv && source .venv/bin/activate

pip install -e ".[dev]"           # editable install + dev extras
# or:
pip install -r requirements.txt
```

## Quickstart

### Run the comprehensive smoke suite (no external API calls)

```bash
python tests/test_smoke_all.py
```

Expected output: `Smoke suite: PASS (0 failures)` across sections S1–S7.

Sections covered:
- **S1** module imports (18 modules)
- **S2** Pydantic schema instantiation (12 types)
- **S3** Critic 6 representative cases (C1, C2, C3, C4, C6, C7)
- **S4** Prediction T1 (dry_run 29 → 87 predictions) + T2 (5-bucket oracle byte-exact, max Δ = 1.11e-16)
- **S5** `boltz_api._extract_usd` (7 cases: numeric, string-form, nested, missing)
- **S6** `graph.py` compiles to `CompiledStateGraph`
- **S7** Reporter emits non-empty markdown report

The original 6-test set is still available via:

```bash
pytest tests/test_smoke.py
```

### Run Research + Structure on PD-L1 (real APIs, no compute key needed)

```python
from peptide_agent.agents import research, structure

state = {"target_id": "PD-L1", "messages": [], "iteration": 0}

state.update(research.run(state))
state.update(structure.run(state))

print(state["epitope_map"].summary)
for h in sorted(state["hotspots"], key=lambda h: h.residue_number):
    print(f"  {h.residue_aa} {h.residue_number:>3d}  role={h.role}  cons={h.conservation:.2f}")
```

Expected output (against 4ZQK):

```
PD-L1 interface on chain A: 22 residues within 5 Å of chain B in PDB 4ZQK;
16 anchors (contact + conserved), 6 rim (contact only). Cross-tool agreement
using 8 mammalian orthologs at ≥88% threshold.
```

### End-to-end CLI (requires `BOLTZ_API_KEY`)

```bash
export BOLTZ_API_KEY=...
python -m peptide_agent --target PD-L1 --max-candidates 29 --n-seeds 3
```

Use `--dry-run` to skip Boltz submissions and exit at the design stage. The
Prediction agent honours a $30 USD cap by default; if the estimate exceeds
that, the agent automatically scopes down to ≤12 candidates × 2 seeds and
emits an `EvidenceCard` recording the decision.

## Evidence ledger

Every external claim — a UniProt entry, a PDB structure, a literature finding,
a conservation score — is stored as an `EvidenceCard`:

```jsonc
{
  "card_id": "evid_25d8ae960b",
  "claim": "PD-L1 canonical sequence is 290 aa (UniProt Q9NZQ7, gene CD274)",
  "source_id": "Q9NZQ7",
  "source_type": "uniprot",
  "source_url": "https://www.uniprot.org/uniprotkb/Q9NZQ7/entry",
  "tag": "VERIFIED",
  "confidence": 0.99,
  "extracted_by": "research",
  "payload": { "length": 290, "gene": "CD274", ... }
}
```

The ledger is append-only and deduplicated by `content_hash` so reruns don't
inflate the trail. Filter helpers:

```python
ledger.by_source("Q9NZQ7")          # exact source-id match
ledger.by_source_type("literature") # all literature cards
```

## Composite scoring (Prediction agent)

The Prediction agent compresses five orthogonal signals into one composite score
for ranking candidates:

| Signal                | Source                        | Weight |
|-----------------------|-------------------------------|--------|
| `ipTM`                | Boltz `iptm_mean`             | 0.40   |
| `hotspot_coverage`    | Predicted vs intended hotspots| 0.20   |
| `cross_tool_agreement`| 1 − \|Boltz iPTM − Chai-1 iPTM\| | 0.15   |
| `interface_pLDDT`     | Mean interface pLDDT / 100    | 0.15   |
| `neg_mean_ddG`        | Clipped (5 − ΔΔG) / 10        | 0.10   |

Missing signals drop their weight and the remainder renormalize. Two override
rules eject a candidate to `rejected`:
- `hotspot_coverage < 0.30` (designed peptide misses the intended hotspots)
- `|Boltz iPTM − Chai-1 iPTM| > 0.25` (cross-tool disagreement above threshold)

Verified byte-exact against the captured PD-L1 oracle (5 buckets, max
Δ = 1.11 × 10⁻¹⁶).

## Reference run: PD-L1

Reproduces under `runs/step2_pdl1_4zqk/`:

- `conservation_q9nzq7.json` — per-position conservation of human PD-L1 against 8 mammalian orthologs (mouse, rat, rabbit, chimp, cow, macaque, pig, dog), computed via Biotite BLOSUM62 pairwise alignment.
- `hotspots_4zqk.json` / `hotspots_4zqk.md` — 22 contact residues at the 4ZQK PD-1/PD-L1 interface (chain A vs B, 5 Å heavy-atom), each tagged anchor / rim based on the 2-axis consensus.
- `evidence_ledger.jsonl` — evidence cards from the run.

## Known caveats

These are real and called out in the code where they bite:

1. **BSA is not computed.** `Hotspot.bsa` is currently `None` for every residue. The interface tool only reports contact, not buried surface area.
2. **Cross-tool agreement is 2-axis, not 3.** PDB contact + mammalian conservation. The original architecture envisioned a third axis (literature mutagenesis residues); that's removed for v0 because the first cut fabricated a curated set that wasn't backed by retrieved papers. The proper fix is to parse residue mentions from `EvidenceCard.payload` data, not to ship a hand-typed list.
3. **`Hotspot.role` Literal still allows `hub` / `ambiguous`** even though the 2-axis pipeline only produces `anchor` and `rim`. Tighten when the third axis lands.
4. **Ortholog selection uses only length filter (±20%).** No `reviewed=true` preference, no phylogenetic balance check. For PD-L1 this returned a clean set; for messier genes it could pick a splice variant or pseudogene.
5. **Critic veto loop in `graph.py` doesn't loop back to Planner.** The edge exists but it's a one-shot, not iterative.
6. **ESM-IF surrogate in the Design agent is a grammar placeholder**, not a real inverse-folding call. Either swap in a genuine inference path or relabel before paper-style claims.

## Contributing

Issues and PRs welcome. See [CONTRIBUTING.md](CONTRIBUTING.md). Please run

```bash
ruff check .
python tests/test_smoke_all.py
```

before opening a PR.

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgements

Project framework guided by the [BioTender](https://biotender.io) OSI-license +
active-maintenance scoring rubric (Junior Yu).
