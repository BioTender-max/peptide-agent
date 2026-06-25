"""LiteratureSearch wrapper.

In the live Phylo runtime, `LiteratureSearch` is exposed as a tool callable
by the LLM, not as a Python function. For PeptideAgent we want callable
research from within agent code, so we use a small adapter that reads the
references.jsonl trail produced by prior LiteratureSearch calls in this
sandbox.

For automated runs that need fresh searches, the run_pdl1 script will
trigger LiteratureSearch from the orchestrator turn and feed results back
to the Research agent.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional


REFERENCES_JSONL = Path("/mnt/results/execution_trace/references.jsonl")


def _load_all_references() -> list[dict]:
    if not REFERENCES_JSONL.exists():
        return []
    out = []
    with REFERENCES_JSONL.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def search_literature(query: str, max_papers: int = 6, year_min: Optional[int] = None) -> list[dict]:
    """Retrieve relevant references previously fetched via LiteratureSearch.

    Lightweight: keyword-substring match against title + claim text.
    For demo this is sufficient because run_pdl1 pre-warms references.jsonl
    via earlier LiteratureSearch tool calls; in v1 we'd embed search.
    """
    refs = _load_all_references()
    q = query.lower()
    hits = []
    for r in refs:
        text = " ".join([
            (r.get("title") or ""),
            (r.get("snippet") or ""),
            (r.get("highlight") or ""),
            (r.get("abstract") or ""),
        ]).lower()
        if any(tok in text for tok in q.split() if len(tok) > 3):
            y = r.get("year")
            if year_min and y and int(y) < year_min:
                continue
            hits.append(r)
            if len(hits) >= max_papers:
                break
    return hits
