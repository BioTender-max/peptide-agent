"""UniProt REST wrapper."""

from __future__ import annotations

import urllib.request
import json


def fetch_uniprot(uniprot_id: str) -> dict:
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
    with urllib.request.urlopen(url, timeout=30) as r:
        data = json.load(r)
    return {
        "uniprot": uniprot_id,
        "gene": data.get("genes", [{}])[0].get("geneName", {}).get("value"),
        "organism": data["organism"]["scientificName"],
        "length": data["sequence"]["length"],
        "sequence": data["sequence"]["value"],
        "function": _get_function(data),
    }


def _get_function(d: dict) -> str:
    for c in d.get("comments", []):
        if c.get("commentType") == "FUNCTION":
            return c.get("texts", [{}])[0].get("value", "")
    return ""
