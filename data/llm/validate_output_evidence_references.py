#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validate cross-field evidence references in GroundwaterSpectra-LLM outputs."""

import argparse
import json
import re
import sys
from pathlib import Path

EVIDENCE_ID = re.compile(r"^EV-[0-9]{3}$")


def load_records(path: Path):
    text = path.read_text(encoding="utf-8-sig").strip()
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    data = json.loads(text)
    return data if isinstance(data, list) else [data]


def validate_record(record):
    well_id = record.get("well_id", "<missing well_id>")
    errors = []

    supporting = record.get("supporting_evidence", [])
    counter = record.get("counter_evidence", [])
    evidence = supporting + counter

    evidence_ids = [item.get("evidence_id") for item in evidence]
    supporting_ids = {item.get("evidence_id") for item in supporting}

    for evidence_id in evidence_ids:
        if not isinstance(evidence_id, str) or not EVIDENCE_ID.fullmatch(evidence_id):
            errors.append(f"{well_id}: invalid evidence_id {evidence_id!r}")

    if len(evidence_ids) != len(set(evidence_ids)):
        errors.append(f"{well_id}: evidence_id values are not unique")

    for index, test in enumerate(record.get("confirmatory_tests", []), start=1):
        refs = test.get("supporting_evidence_ids", [])
        if not refs:
            errors.append(f"{well_id}: confirmatory_tests[{index}] has no supporting_evidence_ids")
        for ref in refs:
            if ref not in supporting_ids:
                errors.append(
                    f"{well_id}: confirmatory_tests[{index}] references undefined "
                    f"supporting evidence {ref!r}"
                )

    all_ids = set(evidence_ids)
    claim_ids = []
    for index, trace in enumerate(record.get("evidence_trace", []), start=1):
        claim_id = trace.get("claim_id")
        claim_ids.append(claim_id)
        refs = trace.get("evidence_ids", [])
        if not refs:
            errors.append(f"{well_id}: evidence_trace[{index}] has no evidence_ids")
        for ref in refs:
            if ref not in all_ids:
                errors.append(
                    f"{well_id}: evidence_trace[{index}] references undefined evidence {ref!r}"
                )

    if len(claim_ids) != len(set(claim_ids)):
        errors.append(f"{well_id}: claim_id values are not unique")

    return errors


def main():
    parser = argparse.ArgumentParser(
        description="Check evidence identifiers and cross-field references in LLM output JSON/JSONL."
    )
    parser.add_argument("output_file", type=Path)
    args = parser.parse_args()

    errors = []
    for record in load_records(args.output_file):
        errors.extend(validate_record(record))

    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    print("Evidence-reference validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
