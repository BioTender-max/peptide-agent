"""Command-line entry point.

Usage:
    python -m peptide_agent --target PD-L1 --out /mnt/results/peptide-agent/runs/pdl1_v0 --dry-run
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

from .graph import build_graph, initial_state


def main():
    p = argparse.ArgumentParser(prog="peptide_agent")
    p.add_argument("--target", default="PD-L1", help="Target identifier (e.g. PD-L1, IL-23R)")
    p.add_argument("--brief", default=None,
                   help="Optional free-text brief; defaults to a template that asks for binder design.")
    p.add_argument("--out", default=None, help="Output directory (default: ./runs/<target>_<timestamp>)")
    p.add_argument("--dry-run", action="store_true",
                   help="Skip Boltz API submission; useful for tests + cost preview.")
    p.add_argument("--max-candidates", type=int, default=None,
                   help="Cap number of candidates predicted (smaller for fast demo).")
    p.add_argument("--n-seeds", type=int, default=3, help="Seeds per Boltz prediction.")
    args = p.parse_args()

    target = args.target
    brief = args.brief or f"Design peptide binders for {target} using structure-guided design."
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out or f"./runs/{target.replace('-','').lower()}_{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    state = initial_state(run_id=out_dir.name, target_id=target, brief=brief, out_dir=str(out_dir))
    state["dry_run"] = args.dry_run
    state["max_candidates"] = args.max_candidates
    state["n_seeds"] = args.n_seeds

    graph = build_graph(checkpoint=True)
    config = {"configurable": {"thread_id": state["run_id"]}}

    # Stream events for visibility
    final = None
    for event in graph.stream(state, config=config, stream_mode="values"):
        final = event
        agent = (event.get("history") or [{}])[-1].get("agent", "?")
        n_ev = len(event.get("evidence_ledger", []))
        print(f"[{agent:>10s}] evidence={n_ev:3d}  candidates={len(event.get('candidates', [])):3d}  scores={len(event.get('scores', {})):3d}")

    if final is None:
        print("Graph produced no events", file=sys.stderr)
        sys.exit(1)
    print(f"\nDone. Report: {final.get('report_path')}")


if __name__ == "__main__":
    main()
