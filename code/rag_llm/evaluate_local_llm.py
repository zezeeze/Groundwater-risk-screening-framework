#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate frozen well-level local LLM outputs against held-out references."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest().upper()


def safe_div(a: float, b: float) -> float | None:
    return a / b if b else None


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def binary_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, Any]:
    tp = sum(t == 1 and p == 1 for t, p in zip(y_true, y_pred))
    tn = sum(t == 0 and p == 0 for t, p in zip(y_true, y_pred))
    fp = sum(t == 0 and p == 1 for t, p in zip(y_true, y_pred))
    fn = sum(t == 1 and p == 0 for t, p in zip(y_true, y_pred))
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    f1 = safe_div(2 * precision * recall, precision + recall) if precision is not None and recall is not None else None
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = safe_div(tp * tn - fp * fn, denominator)
    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": safe_div(tp + tn, len(y_true)), "precision": precision,
        "recall": recall, "specificity": specificity, "f1": f1, "mcc": mcc,
    }


def topk_metrics(predicted: list[list[str]], references: list[set[str]], k: int) -> dict[str, Any]:
    hits, recalls, precisions = [], [], []
    for pred, ref in zip(predicted, references):
        chosen = pred[:k]
        overlap = len(set(chosen) & ref)
        hits.append(float(overlap > 0))
        recalls.append(overlap / len(ref))
        precisions.append(overlap / len(chosen) if chosen else 0.0)
    return {
        "n_evaluable_wells": len(references),
        "any_hit_rate": mean(hits),
        "macro_recall": mean(recalls),
        "macro_precision": mean(precisions),
    }


def enterprise_metrics(rows: list[tuple[list[str], set[str]]], k: int) -> dict[str, Any]:
    if not rows:
        return {"n_evaluable_wells": 0, "any_hit_rate": None, "macro_recall": None, "macro_precision": None}
    return topk_metrics([x[0] for x in rows], [x[1] for x in rows], k)


POLLUTANT_PATTERNS = [
    ("1,2-二氯乙烷", "1,2-Dichloroethane", "特征有机污染物"),
    ("二氯甲烷", "Dichloromethane", "特征有机污染物"),
    ("三氯甲烷", "Chloroform", "特征有机污染物"),
    ("氯乙烯", "Vinyl chloride", "特征有机污染物"),
    ("乙醛", "Acetaldehyde", "特征有机污染物"),
    ("硫酸盐", "Sulfate", "盐分与矿化度"),
    ("氯化物", "Chloride", "盐分与矿化度"),
    ("耗氧量", "CODMn", "常规水质与营养盐"),
    ("氨氮", "Ammonia-N", "常规水质与营养盐"),
    ("硝酸盐", "Nitrate-N", "常规水质与营养盐"),
    ("亚硝酸盐", "Nitrite-N", "常规水质与营养盐"),
    ("pH", "pH", "常规水质与营养盐"),
    ("Mn", "Mn", "常见金属"), ("锰", "Mn", "常见金属"),
    ("Al", "Al", "常见金属"), ("铝", "Al", "常见金属"),
    ("Fe", "Fe", "常见金属"), ("铁", "Fe", "常见金属"),
    ("Cu", "Cu", "毒性金属与类金属"), ("铜", "Cu", "毒性金属与类金属"),
    ("Zn", "Zn", "毒性金属与类金属"), ("锌", "Zn", "毒性金属与类金属"),
    ("Cd", "Cd", "毒性金属与类金属"), ("镉", "Cd", "毒性金属与类金属"),
    ("Ni", "Ni", "毒性金属与类金属"), ("镍", "Ni", "毒性金属与类金属"),
    ("Co", "Co", "毒性金属与类金属"), ("钴", "Co", "毒性金属与类金属"),
    ("As", "As", "毒性金属与类金属"), ("砷", "As", "毒性金属与类金属"),
    ("Pb", "Pb", "毒性金属与类金属"), ("铅", "Pb", "毒性金属与类金属"),
    ("Tl", "Tl", "毒性金属与类金属"), ("铊", "Tl", "毒性金属与类金属"),
]


def canonical_pollutant(value: str) -> str:
    text = str(value).strip()
    for pattern, canonical, _ in POLLUTANT_PATTERNS:
        if text.casefold() == pattern.casefold():
            return canonical
    return re.sub(r"\s+", "", text).casefold()


def reference_link_sets(reference_items: list[dict[str, Any]]) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    strict: set[tuple[str, str]] = set()
    category: set[tuple[str, str]] = set()
    for item in reference_items:
        enterprise_id = str(item.get("enterprise_id", ""))
        direction = str(item.get("pollutant_direction", ""))
        for pattern, canonical, risk_category in POLLUTANT_PATTERNS:
            if re.search(rf"(?<![A-Za-z]){re.escape(pattern)}(?![A-Za-z])", direction, flags=re.IGNORECASE):
                strict.add((enterprise_id, canonical))
                category.add((enterprise_id, risk_category))
    return strict, category


def pair_metrics(predicted: list[list[tuple[str, str]]], references: list[set[tuple[str, str]]], k: int) -> dict[str, Any]:
    hits, recalls = [], []
    for pred, ref in zip(predicted, references):
        chosen = set(pred[:k])
        overlap = len(chosen & ref)
        hits.append(float(overlap > 0))
        recalls.append(overlap / len(ref))
    return {"n_evaluable_wells": len(references), "any_hit_rate": mean(hits), "macro_recall": mean(recalls)}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def validate_structured_output(record: dict[str, Any]) -> dict[str, bool]:
    """Check only deterministic schema fields and cross-field evidence links."""
    required_fields = {
        "well_id", "model_status", "risk_screening_level", "overall_assessment",
        "supporting_evidence", "counter_evidence", "priority_risk_categories",
        "confirmatory_tests", "contaminant_enterprise_links", "suspected_enterprises", "quality_control_checks",
        "unexcluded_risks", "uncertainty", "evidence_trace",
    }
    required_fields_complete = required_fields.issubset(record)

    supporting = record.get("supporting_evidence", [])
    counter = record.get("counter_evidence", [])
    evidence_ids = {
        item.get("evidence_id")
        for item in supporting + counter
        if isinstance(item, dict) and item.get("evidence_id")
    }
    supporting_ids = {
        item.get("evidence_id")
        for item in supporting
        if isinstance(item, dict) and item.get("evidence_id")
    }
    test_refs = [
        ref
        for item in record.get("confirmatory_tests", [])
        if isinstance(item, dict)
        for ref in item.get("supporting_evidence_ids", [])
    ]
    trace_refs = [
        ref
        for item in record.get("evidence_trace", [])
        if isinstance(item, dict)
        for ref in item.get("evidence_ids", [])
    ]
    evidence_references_valid = (
        all(ref in supporting_ids for ref in test_refs)
        and all(ref in evidence_ids for ref in trace_refs)
    )
    return {
        "required_fields_complete": required_fields_complete,
        "evidence_references_valid": evidence_references_valid,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    freeze = json.loads(args.freeze_manifest.read_text(encoding="utf-8-sig"))
    current_hash = sha256(args.outputs)
    if current_hash != freeze["sha256"]:
        raise RuntimeError("Frozen output hash mismatch; evaluation aborted")

    outputs = read_jsonl(args.outputs)
    reference_doc = json.loads(args.references.read_text(encoding="utf-8-sig"))
    references = reference_doc["records"]
    inputs = read_jsonl(args.inputs)
    omap = {x["well_id"]: x for x in outputs}
    rmap = {x["well_id"]: x for x in references}
    imap = {
        x["well_id"]: x.get("well_evidence_package", x)
        for x in inputs
    }
    if set(omap) != set(rmap):
        raise RuntimeError(f"Well mismatch: missing={set(rmap)-set(omap)}, extra={set(omap)-set(rmap)}")

    well_rows = []
    y_true, y_pred = [], []
    category_pred, category_ref = [], []
    test_pred, test_ref = [], []
    enterprise_all, enterprise_risk = [], []
    enterprise_distance_all, enterprise_distance_risk = [], []
    strict_pair_pred, strict_pair_ref = [], []
    category_pair_pred, category_pair_ref = [], []
    for well_id in rmap:
        out, ref, inp = omap[well_id], rmap[well_id], imap[well_id]
        predicted_risk = int(out["model_status"] != "Both-low")
        actual_risk = int(ref["overall_risk_label"])
        y_true.append(actual_risk)
        y_pred.append(predicted_risk)
        categories = [x["category"] for x in sorted(out["priority_risk_categories"], key=lambda x: x["rank"])]
        tests = [x["indicator"] for x in sorted(out["confirmatory_tests"], key=lambda x: x["priority"])]
        enterprises = [x["enterprise_id"] for x in sorted(out["suspected_enterprises"], key=lambda x: x["rank"])]
        distance_enterprises = [
            x["enterprise_id"] for x in sorted(
                inp["enterprise_context"]["candidate_enterprises"], key=lambda x: x["distance_rank"]
            )
        ]
        links = sorted(out.get("contaminant_enterprise_links", []), key=lambda x: x["rank"])
        predicted_strict_pairs = [
            (x["enterprise_id"], canonical_pollutant(x["candidate_contaminant"])) for x in links
        ]
        predicted_category_pairs = [(x["enterprise_id"], x["risk_category"]) for x in links]
        positive_categories = set(ref["positive_categories"])
        accepted_tests = set(ref["accepted_confirmatory_test_targets"])
        enterprise_refs = {x["enterprise_id"] for x in ref["enterprise_reference"]}
        if positive_categories:
            category_pred.append(categories)
            category_ref.append(positive_categories)
        if accepted_tests:
            test_pred.append(tests)
            test_ref.append(accepted_tests)
        if enterprise_refs:
            enterprise_all.append((enterprises, enterprise_refs))
            enterprise_distance_all.append((distance_enterprises, enterprise_refs))
            if actual_risk:
                enterprise_risk.append((enterprises, enterprise_refs))
                enterprise_distance_risk.append((distance_enterprises, enterprise_refs))
        reference_strict_pairs, reference_category_pairs = reference_link_sets(ref["enterprise_reference"])
        if reference_strict_pairs:
            strict_pair_pred.append(predicted_strict_pairs)
            strict_pair_ref.append(reference_strict_pairs)
        if reference_category_pairs:
            category_pair_pred.append(predicted_category_pairs)
            category_pair_ref.append(reference_category_pairs)
        well_rows.append({
            "well_id": well_id,
            "overall_risk_reference": actual_risk,
            "overall_risk_prediction": predicted_risk,
            "model_status": out["model_status"],
            "risk_screening_level": out["risk_screening_level"],
            "positive_categories": ";".join(ref["positive_categories"]),
            "ranked_categories": ";".join(categories),
            "category_top1_hit": int(bool(set(categories[:1]) & positive_categories)) if positive_categories else "",
            "category_top3_hit": int(bool(set(categories[:3]) & positive_categories)) if positive_categories else "",
            "category_top5_hit": int(bool(set(categories[:5]) & positive_categories)) if positive_categories else "",
            "accepted_tests": ";".join(ref["accepted_confirmatory_test_targets"]),
            "recommended_tests": ";".join(tests),
            "test_top1_hit": int(bool(set(tests[:1]) & accepted_tests)) if accepted_tests else "",
            "test_top3_hit": int(bool(set(tests[:3]) & accepted_tests)) if accepted_tests else "",
            "test_top5_hit": int(bool(set(tests[:5]) & accepted_tests)) if accepted_tests else "",
            "test_top10_hit": int(bool(set(tests[:10]) & accepted_tests)) if accepted_tests else "",
            "enterprise_reference": ";".join(sorted(enterprise_refs)),
            "enterprise_predictions": ";".join(enterprises),
            "distance_only_enterprises": ";".join(distance_enterprises[:5]),
            "contaminant_enterprise_links": ";".join(f"{x['enterprise_id']}|{x['candidate_contaminant']}|{x['risk_category']}" for x in links),
            "pair_strict_top5_hit": int(bool(set(predicted_strict_pairs[:5]) & reference_strict_pairs)) if reference_strict_pairs else "",
            "pair_category_top5_hit": int(bool(set(predicted_category_pairs[:5]) & reference_category_pairs)) if reference_category_pairs else "",
            "pair_strict_top10_hit": int(bool(set(predicted_strict_pairs[:10]) & reference_strict_pairs)) if reference_strict_pairs else "",
            "pair_category_top10_hit": int(bool(set(predicted_category_pairs[:10]) & reference_category_pairs)) if reference_category_pairs else "",
            "enterprise_top1_hit": int(bool(set(enterprises[:1]) & enterprise_refs)) if enterprise_refs else "",
            "enterprise_top3_hit": int(bool(set(enterprises[:3]) & enterprise_refs)) if enterprise_refs else "",
            "enterprise_top5_hit": int(bool(set(enterprises[:5]) & enterprise_refs)) if enterprise_refs else "",
        })

    overall = binary_metrics(y_true, y_pred)
    topk_rows = []
    category_summary, test_summary = {}, {}
    for k in (1, 3):
        category_summary[f"top_{k}"] = topk_metrics(category_pred, category_ref, k)
        topk_rows.append({"task": "risk_category", "k": k, **category_summary[f"top_{k}"]})
    for k in (3, 5, 10):
        test_summary[f"top_{k}"] = topk_metrics(test_pred, test_ref, k)
        topk_rows.append({"task": "confirmatory_test", "k": k, **test_summary[f"top_{k}"]})

    enterprise_summary = {
        "full_evidence_all_reference_wells": {f"top_{k}": enterprise_metrics(enterprise_all, k) for k in (1, 3, 5)},
        "distance_only_all_reference_wells": {f"top_{k}": enterprise_metrics(enterprise_distance_all, k) for k in (1, 3, 5)},
        "full_evidence_risk_reference_wells": {f"top_{k}": enterprise_metrics(enterprise_risk, k) for k in (1, 3, 5)},
        "distance_only_risk_reference_wells": {f"top_{k}": enterprise_metrics(enterprise_distance_risk, k) for k in (1, 3, 5)},
    }
    link_summary = {
        "strict_pollutant_enterprise_match": {f"top_{k}": pair_metrics(strict_pair_pred, strict_pair_ref, k) for k in (3, 5, 10)},
        "category_tolerant_match": {f"top_{k}": pair_metrics(category_pair_pred, category_pair_ref, k) for k in (3, 5, 10)},
        "category_tolerant_definition": "Exact enterprise plus a specific candidate contaminant mapped to the same predefined risk category as at least one explicit contaminant in the independent reference. Generic metal, organic or inorganic phrases do not create a match.",
    }

    structural_checks = [validate_structured_output(record) for record in outputs]
    structural_validation = {
        "n_outputs": len(outputs),
        "required_field_completeness_rate": mean([
            float(item["required_fields_complete"]) for item in structural_checks
        ]),
        "evidence_reference_validity_rate": mean([
            float(item["evidence_references_valid"]) for item in structural_checks
        ]),
        "note": "Deterministic checks of required fields and evidence identifiers; no semantic grounding or expert-quality score is inferred.",
    }

    summary = {
        "evaluation_after_output_freeze": True,
        "frozen_output_sha256": current_hash,
        "n_wells": len(outputs),
        "overall_risk": overall,
        "risk_category_ranking": category_summary,
        "confirmatory_test_recommendation": test_summary,
        "enterprise_candidate_identification": enterprise_summary,
        "contaminant_enterprise_link_identification": link_summary,
        "structured_output_validation": structural_validation,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "evaluation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(args.output_dir / "well_level_metrics.csv", well_rows)
    write_csv(args.output_dir / "category_and_test_topk_metrics.csv", topk_rows)
    write_csv(args.output_dir / "enterprise_candidate_metrics.csv", [
        {"scope": scope, "k": k, **metrics}
        for scope, values in enterprise_summary.items()
        for key, metrics in values.items()
        for k in [int(key.split("_")[-1])]
    ])
    write_csv(args.output_dir / "contaminant_enterprise_metrics.csv", [
        {"match_rule": rule, "k": k, **metrics}
        for rule, values in link_summary.items() if isinstance(values, dict)
        for key, metrics in values.items()
        for k in [int(key.split("_")[-1])]
    ])
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
