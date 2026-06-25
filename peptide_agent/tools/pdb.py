"""PDB / RCSB REST wrapper + file fetcher."""

from __future__ import annotations

import os
import urllib.request
import json
from pathlib import Path


PDB_CACHE = Path(os.environ.get("PDB_CACHE", "/workspace/pdb_cache"))
PDB_CACHE.mkdir(parents=True, exist_ok=True)


def fetch_pdb_summary(pdb_id: str) -> dict:
    pdb_id = pdb_id.upper()
    url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
    with urllib.request.urlopen(url, timeout=30) as r:
        d = json.load(r)
    return {
        "pdb_id": pdb_id,
        "title": (d.get("struct", {}).get("title") or "").strip(),
        "method": (d.get("exptl", [{}])[0].get("method") or "").strip(),
        "resolution": (d.get("rcsb_entry_info", {}).get("resolution_combined") or [None])[0],
        "deposited": (d.get("rcsb_accession_info", {}) or {}).get("deposit_date"),
        "polymer_entity_count": d.get("rcsb_entry_info", {}).get("polymer_entity_count"),
    }


def fetch_pdb_file(pdb_id: str) -> str:
    pdb_id = pdb_id.upper()
    cached = PDB_CACHE / f"{pdb_id}.cif"
    if cached.exists() and cached.stat().st_size > 0:
        return str(cached)
    url = f"https://files.rcsb.org/download/{pdb_id}.cif"
    with urllib.request.urlopen(url, timeout=60) as r:
        data = r.read()
    cached.write_bytes(data)
    return str(cached)
