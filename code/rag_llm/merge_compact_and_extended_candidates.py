#!/usr/bin/env python3
"""Preserve the compact Top-5 list and append unique extended candidates to Top-10."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

from run_local_qwen import (
    load_method_lookup,
    normalize_deterministic_fields,
    read_jsonl,
    write_jsonl,
)


def merge_unique(compact: list[dict], extended: list[dict], key, limit: int) -> list[dict]:
    merged: list[dict] = []
    seen = set()
    for item in [*compact, *extended]:
        item_key = key(item)
        if item_key in seen:
            continue
        seen.add(item_key)
        merged.append(deepcopy(item))
        if len(merged) == limit:
            break
    return merged


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--compact", type=Path, required=True)
    parser.add_argument("--extended", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import json
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    requests = {x["well_id"]: x for x in read_jsonl(Path(config["request_file"]))}
    compact = {x["well_id"]: x for x in read_jsonl(args.compact)}
    extended = {x["well_id"]: x for x in read_jsonl(args.extended)}
    method_lookup = load_method_lookup(Path(config["corpus_file"]))
    outputs = []

    for well_id, request in requests.items():
        result = deepcopy(compact[well_id])
        result["confirmatory_tests"] = merge_unique(
            sorted(compact[well_id]["confirmatory_tests"], key=lambda x: x["priority"]),
            sorted(extended[well_id]["confirmatory_tests"], key=lambda x: x["priority"]),
            key=lambda x: str(x["indicator"]).casefold(),
            limit=10,
        )
        result["contaminant_enterprise_links"] = merge_unique(
            sorted(compact[well_id]["contaminant_enterprise_links"], key=lambda x: x["rank"]),
            sorted(extended[well_id]["contaminant_enterprise_links"], key=lambda x: x["rank"]),
            key=lambda x: (str(x["enterprise_id"]), str(x["candidate_contaminant"]).casefold()),
            limit=10,
        )
        result = normalize_deterministic_fields(result, request, method_lookup)
        outputs.append(result)

    write_jsonl(args.output, outputs)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
