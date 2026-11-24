"""Prospective evaluation revision. Never mix changed prompts/data with old predictions."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

VERSION = "2026-10-23-v2"
HERE = Path(__file__).resolve().parent
DEFAULT_PREDS = str(HERE / "preds_v2")


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def fingerprint(items, settings, source_files=()):
    sources = {str(Path(p).resolve()): hashlib.sha256(Path(p).read_bytes()).hexdigest()
               for p in source_files}
    return digest({"version": VERSION, "items": items, "settings": settings, "sources": sources})


def resume_ids(path, run_fingerprint):
    """Reject stale or corrupt outputs instead of silently skipping IDs."""
    done = set()
    if not Path(path).exists():
        return done
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("run_fingerprint") != run_fingerprint:
                raise ValueError(f"Protocol/data/settings changed: use a new --out directory ({path})")
            if not rec.get("error"):
                done.add(rec["id"])
    return done


def stamp(rec, run_fingerprint):
    rec.update(protocol_version=VERSION, run_fingerprint=run_fingerprint)
    return rec


def exclude_items(items, paths):
    """Pilot JSONL/one-ID-per-line lists; claim-level grouping where available."""
    excluded = set()
    for path in paths or []:
        with open(path, encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                value = json.loads(line) if line.lstrip().startswith("{") else {"id": line.strip()}
                excluded.add(str(value["id"]))
                cid = (value.get("meta") or {}).get("claim_id")
                if cid is not None:
                    excluded.add("claim:" + str(cid))
    return [r for r in items if str(r["id"]) not in excluded and
            "claim:" + str((r.get("meta") or {}).get("claim_id")) not in excluded]
