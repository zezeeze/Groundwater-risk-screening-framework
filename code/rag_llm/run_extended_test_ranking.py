#!/usr/bin/env python3
"""Rank an extended ten-item confirmatory screening panel for each well."""

from __future__ import annotations

import argparse
import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

from run_local_qwen import (
    load_method_lookup,
    normalize_deterministic_fields,
    post_completion,
    read_jsonl,
    write_jsonl,
)


TEST_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["tests"],
    "properties": {
        "tests": {
            "type": "array",
            "minItems": 10,
            "maxItems": 10,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["rank", "indicator"],
                "properties": {
                    "rank": {"type": "integer", "minimum": 1, "maximum": 10},
                    "indicator": {"type": "string"},
                },
            },
        }
    },
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def compact_record(
    request: dict[str, Any],
    base_output: dict[str, Any],
    method_lookup: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    package = request["well_evidence_package"]
    categories = []
    category_score: dict[str, float] = {}
    definitions = {
        item["category_name_zh"]: item
        for item in request["shared_category_model_performance"]["category_definitions"]
    }
    for item in package["risk_category_models"]:
        name = item["category_name_zh"]
        eem = float(item["eem"]["probability"])
        uvvis = float(item["uvvis"]["probability"])
        category_score[name] = max(eem, uvvis)
        indicators = []
        for indicator in definitions[name]["included_indicators"]:
            method = method_lookup.get(str(indicator).casefold())
            if method and method["indicator_id"] not in indicators:
                indicators.append(method["indicator_id"])
        categories.append({
            "category": name,
            "eem_probability": eem,
            "uvvis_probability": uvvis,
            "candidate_indicators": indicators,
        })

    selected_enterprises = {
        item["enterprise_id"] for item in base_output.get("suspected_enterprises", [])
    }
    enterprise_pollutants = []
    for item in package.get("enterprise_context", {}).get("candidate_enterprises", []):
        if item.get("enterprise_id") not in selected_enterprises:
            continue
        enterprise_pollutants.append({
            "enterprise_id": item["enterprise_id"],
            "reported_pollutants": item.get("production_alignment", {}).get(
                "matched_reported_pollutants", []
            ),
        })
    return {
        "well_id": request["well_id"],
        "overall_model_status": package["overall_risk_models"]["model_status"],
        "risk_category_evidence": categories,
        "selected_enterprise_reported_pollutants": enterprise_pollutants,
        "category_score": category_score,
    }


def build_prompt(record: dict[str, Any]) -> str:
    payload = {k: v for k, v in record.items() if k != "category_score"}
    return (
        "<|im_start|>system\n/no_think\n"
        "你负责为工业园区地下水初筛组织扩展确认性检测清单。"
        "只能从输入的candidate_indicators中选择项目，不得补充名单外项目。"
        "排序需综合两条光谱路径的类别概率、总体模型状态及候选企业记录的特征污染指标。"
        "该清单用于初筛后续核查，应兼顾高优先级方向与不同化学类别的覆盖。<|im_end|>\n"
        "<|im_start|>user\n"
        "请返回10个不重复检测项目并排序。只能返回符合JSON Schema的对象，不写解释性正文。\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        + "\n<|im_end|>\n<|im_start|>assistant\n/no_think\n"
    )


def normalize_tests(
    generated: list[dict[str, Any]],
    record: dict[str, Any],
) -> list[str]:
    allowed_by_category = {
        item["category"]: item["candidate_indicators"]
        for item in record["risk_category_evidence"]
    }
    allowed = {
        indicator.casefold(): indicator
        for values in allowed_by_category.values()
        for indicator in values
    }
    ranked: list[str] = []
    seen: set[str] = set()
    for item in sorted(generated, key=lambda x: int(x.get("rank", 999))):
        indicator = allowed.get(str(item.get("indicator", "")).casefold())
        if not indicator or indicator.casefold() in seen:
            continue
        seen.add(indicator.casefold())
        ranked.append(indicator)

    category_order = sorted(
        allowed_by_category,
        key=lambda category: -record["category_score"].get(category, 0.0),
    )
    for category in category_order:
        for indicator in allowed_by_category[category]:
            if indicator.casefold() in seen:
                continue
            seen.add(indicator.casefold())
            ranked.append(indicator)
            if len(ranked) == 10:
                return ranked
    return ranked[:10]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--base-outputs", required=True)
    args = parser.parse_args()
    config = load_json(Path(args.config))
    requests = read_jsonl(Path(config["request_file"]))
    base = {x["well_id"]: x for x in read_jsonl(Path(args.base_outputs))}
    method_lookup = load_method_lookup(Path(config["corpus_file"]))
    output_dir = Path(config["output_dir"])
    output_path = output_dir / "all_wells_qwen3_8b_extended_tests.jsonl"
    raw_path = output_dir / "all_wells_qwen3_8b_extended_tests_raw_responses.jsonl"
    log_path = output_dir / "all_wells_qwen3_8b_extended_tests_run_log.json"
    outputs: list[dict[str, Any]] = []
    raw_records: list[dict[str, Any]] = []
    logs: list[dict[str, Any]] = []
    compact_config = deepcopy(config)
    compact_config["max_output_tokens"] = 360
    compact_config["max_attempts"] = 1

    for index, request in enumerate(requests, start=1):
        well_id = request["well_id"]
        record = compact_record(request, base[well_id], method_lookup)
        started = time.perf_counter()
        response = post_completion(config["endpoint"], build_prompt(record), TEST_SCHEMA, compact_config)
        elapsed = time.perf_counter() - started
        raw_records.append({"well_id": well_id, "response": response})
        try:
            generated = json.loads(response["content"]).get("tests", [])
            status = "parsed"
        except json.JSONDecodeError:
            generated = []
            status = "fallback_after_parse_error"
        indicators = normalize_tests(generated, record)
        merged = deepcopy(base[well_id])
        merged["confirmatory_tests"] = [
            {
                "indicator": indicator,
                "priority": rank,
                "purpose": "用于扩展筛查清单中的候选异常核查",
                "supporting_evidence_ids": ["EV-001", "EV-002"],
                "method_or_qc_note": "",
            }
            for rank, indicator in enumerate(indicators, start=1)
        ]
        merged = normalize_deterministic_fields(merged, request, method_lookup)
        outputs.append(merged)
        logs.append({
            "well_id": well_id,
            "status": status,
            "elapsed_seconds": round(elapsed, 3),
            "tokens_evaluated": response.get("tokens_evaluated"),
            "tokens_predicted": response.get("tokens_predicted"),
            "generated_items": len(generated),
            "final_items": len(merged["confirmatory_tests"]),
        })
        write_jsonl(raw_path, raw_records)
        write_jsonl(output_path, outputs)
        log_path.write_text(json.dumps({"items": logs}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[{index}/{len(requests)}] {well_id}: {status}, {elapsed:.1f}s", flush=True)

    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
