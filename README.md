<div align="center">

# 🧬 PeptideAgent

**Agentic peptide-binder discovery with a tracked evidence trail.**

[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-multi--agent-1C3A5E?style=flat-square)](https://langchain-ai.github.io/langgraph/)
[![License: MIT](https://img.shields.io/badge/license-MIT-22c55e?style=flat-square)](LICENSE)
[![Status](https://img.shields.io/badge/status-alpha-f59e0b?style=flat-square)]()
[![Tests](https://img.shields.io/badge/smoke_tests-6%2F6_passing-22c55e?style=flat-square)](tests/)

A [LangGraph](https://langchain-ai.github.io/langgraph/)-based **7-agent pipeline** that takes a protein target + epitope brief and produces fully-cited peptide binder candidates — with every claim backed by a verifiable `EvidenceCard`.

</div>

---

## How it works

```
Target + Epitope Brief
        │
        ▼
  ┌─────────────┐
  │   Planner   │  ← decomposes task, assigns subtasks
  └──────┬──────┘
         │
    ┌────┴────┐
    ▼         ▼
┌──────────┐ ┌───────────┐
│ Research │ │ Structure │  ← UniProt · RCSB · Literature
└────┬─────┘ └─────┬─────┘    Biotite · Conservation
     └──────┬───────┘
            ▼
      ┌───────────┐
      │  Designer  │  ← mutation scan · ESM-IF · cyclic hairpin
      └─────┬─────┘
            ▼
      ┌────────────┐
      │ Predictor  │  ← Boltz structure prediction API
      └─────┬──────┘
            ▼
      ┌────────┐
      │ Critic │  ← 4-layer veto gate (≥2 independent sources required)
      └────┬───┘
           ▼
      ┌──────────┐
      │ Reporter │  ← Markdown report + evidence ledger
      └──────────┘
```

The architectural focus is **provenance**: every claim a downstream agent uses is backed by an `EvidenceCard` with a `source_id`, `source_url`, `tag` (VERIFIED / DERIVED / SUBJECTIVE), and a `confidence` score. The Critic vetoes anything not backed by ≥2 independent tools.

---

## Status

> Alpha research code. Numbers reflect the demo run on the reference case (PD-L1).

| Subsystem | State | Validation |
|-----------|-------|------------|
| Scaffold (schemas, ledger, graph, supervisor) | ✅ Done | 6/6 smoke tests passing |
| Research agent (UniProt + RCSB + literature) | ✅ Real APIs | PD-L1: 13 evidence cards in ~3s |
| Structure agent (Biotite interface + conservation) | ✅ Real APIs | 4ZQK: 22 contact residues, 2-axis consensus |
| Design agent (mutation scan + ESM-IF + cyclic hairpin) | 🟡 Scaffold | Untested against rewritten Structure output |
| Prediction agent (Boltz Compute API) | 🟡 Scaffold | Boltz request shape not yet exercised |
| Critic (4-layer gate) | 🟡 Scaffold | Veto-loop edge not yet reactive |
| Reporter (Markdown render) | 🟡 Scaffold | Untested |

---

## Install

```bash
git clone https://github.com/BioTender-max/peptide-agent.git
cd peptide-agent

python -m venv .venv && source .venv/bin/activate   # Python ≥ 3.10

pip install -e ".[dev]"       # editable + dev extras
# or:
pip install -r requirements.txt
```

---

## Quickstart

### Smoke tests (no external API calls)

```bash
pytest tests/
# Expected: 6 passed
```

### Research + Structure on PD-L1 (free APIs, no key needed)

```python
from peptide_agent.agents import research, structure

state = {"target_id": "PD-L1", "messages": [], "iteration": 0}
state.update(research.run(state))
state.update(structure.run(state))

print(state["epitope_map"].summary)
for h in sorted(state["hotspots"], key=lambda h: h.residue_number):
    print(f"  {h.residue_aa}{h.residue_number:>3d}  role={h.role}  cons={h.conservation:.2f}")
```

**Expected output (4ZQK):**
```
PD-L1 interface on chain A: 22 residues within 5 Å of chain B in PDB 4ZQK;
16 anchors (contact + conserved), 6 rim (contact only).
```

### End-to-end CLI (requires `BOLTZ_API_KEY`)

```bash
export BOLTZ_API_KEY=...
python -m peptide_agent --target PD-L1 --max-candidates 5 --n-seeds 2

# Skip Boltz submissions, exit at design stage:
python -m peptide_agent --target PD-L1 --dry-run
```

---

## Project layout

```
peptide_agent/
├── __main__.py          # CLI entry point
├── schemas.py           # Pydantic types: EvidenceCard, TaskPlan, EpitopeMap …
├── state.py             # LangGraph AgentState TypedDict
├── graph.py             # LangGraph composition with conditional edges
├── agents/              # 7 agents + supervisor router
│   ├── planner.py       ├── research.py    ├── structure.py
│   ├── design.py        ├── prediction.py  ├── critic.py
│   ├── reporter.py      └── supervisor.py
├── tools/               # Tool wrappers (all external calls go through here)
│   ├── uniprot.py       ├── pdb.py         ├── literature.py
│   ├── interface.py     ├── conservation.py └── boltz_api.py
├── ledger/store.py      # Append-only EvidenceLedger w/ content-hash dedup
└── reporting/           # Figure + report rendering helpers
tests/test_smoke.py
runs/                    # Per-run outputs
cache/                   # Tool caches (conservation_<accession>.json)
docs/                    # Architecture docs
```

---

## Evidence ledger

Every external claim — UniProt entry, PDB structure, literature finding, conservation score — is stored as an `EvidenceCard`:

```jsonc
{
  "card_id": "evid_25d8ae960b",
  "claim": "PD-L1 canonical sequence is 290 aa (UniProt Q9NZQ7, gene CD274)",
  "source_id": "Q9NZQ7",
  "source_type": "uniprot",
  "source_url": "https://www.uniprot.org/uniprotkb/Q9NZQ7/entry",
  "tag": "VERIFIED",
  "confidence": 0.99,
  "extracted_by": "research"
}
```

The ledger is append-only and deduplicated by `content_hash`. Filter helpers:

```python
ledger.by_source("Q9NZQ7")           # exact source-id match
ledger.by_source_type("literature")  # all literature cards
```

---

## Reference run: PD-L1

Outputs under [`runs/step2_pdl1_4zqk/`](runs/step2_pdl1_4zqk/):

| File | Description |
|------|-------------|
| `conservation_q9nzq7.json` | Per-position conservation of human PD-L1 against 8 mammalian orthologs via BLOSUM62 |
| `hotspots_4zqk.json` | 22 contact residues at the 4ZQK PD-1/PD-L1 interface, tagged anchor/rim |
| `hotspots_4zqk.md` | Human-readable hotspot summary |
| `evidence_ledger.jsonl` | 37 evidence cards |

---

## Known limitations

| # | Issue | Impact |
|---|-------|--------|
| 1 | **BSA not computed** — `Hotspot.bsa` is `None` | Interface tool reports contact only, not buried surface area |
| 2 | **2-axis agreement only** — literature mutagenesis axis removed in v0 (fabricated residues) | Third axis planned for v1 |
| 3 | **Critic veto loop is one-shot** — graph edge exists but doesn't iterate back to Planner | Not yet reactive |
| 4 | **Ortholog selection uses length filter only** (±20%), no phylogenetic balance | May pick splice variants for messy genes |

---

## Contributing

Issues and PRs welcome. Run before opening a PR:

```bash
pytest tests/
ruff check .
```

---

## License

MIT — see [LICENSE](LICENSE).

---

<div align="center">

Built with [LangGraph](https://langchain-ai.github.io/langgraph/) · [UniProt](https://www.uniprot.org/) · [RCSB PDB](https://www.rcsb.org/) · [Boltz](https://github.com/jwohlwend/boltz)

Guided by the [BioTender](https://www.biotender.online/) AI-for-Biology scoring rubric.

</div>
