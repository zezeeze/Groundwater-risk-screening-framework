#!/usr/bin/env python3
"""Freeze an LLM JSONL output before independent reference evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.outputs.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    well_ids = [str(row.get("well_id", "")) for row in rows]
    manifest = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_file": args.outputs.as_posix(),
        "sha256": sha256(args.outputs),
        "n_records": len(rows),
        "unique_wells": len(set(well_ids)),
        "independent_evaluation_answers_loaded_before_freeze": False,
        "note": "Output frozen before independent reference evaluation.",
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
