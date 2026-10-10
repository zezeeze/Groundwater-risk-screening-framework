#!/usr/bin/env python3
"""Run the compact contaminant-enterprise linkage stage and merge it into frozen well outputs."""

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


LINK_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["links"],
    "properties": {
        "links": {
            "type": "array",
            "minItems": 10,
            "maxItems": 10,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["rank", "enterprise_id", "candidate_contaminant", "risk_category"],
                "properties": {
                    "rank": {"type": "integer", "minimum": 1, "maximum": 10},
                    "enterprise_id": {"type": "string"},
                    "candidate_contaminant": {"type": "string"},
                    "risk_category": {
                        "enum": [
                            "常规水质与营养盐",
                            "盐分与矿化度",
                            "常见金属",
                            "毒性金属与类金属",
                            "特征有机污染物",
                        ]
                    },
                },
            },
        }
    },
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def compact_record(request: dict[str, Any]) -> dict[str, Any]:
    package = request["well_evidence_package"]
    categories = []
    category_score: dict[str, float] = {}
    for item in package["risk_category_models"]:
        name = item["category_name_zh"]
        eem = float(item["eem"]["probability"])
        uvvis = float(item["uvvis"]["probability"])
        category_score[name] = max(eem, uvvis)
        categories.append({"category": name, "eem_probability": eem, "uvvis_probability": uvvis})
    candidates = []
    for item in package["enterprise_context"]["candidate_enterprises"]:
        mappings = item.get("production_alignment", {}).get("reported_pollutant_category_map", [])
        if not mappings:
            continue
        candidates.append({
            "enterprise_id": item["enterprise_id"],
            "distance_rank": item.get("distance_rank"),
            "distance_band": item.get("distance_band"),
            "enterprise_relative_direction": item.get("enterprise_relative_direction"),
            "regional_hydraulic_consistency": item.get("regional_hydraulic_consistency"),
            "operating_status": item.get("operating_status"),
            "industry": item.get("industry"),
            "products": str(item.get("products", ""))[:180],
            "raw_materials": str(item.get("raw_materials", ""))[:180],
            "reported_pollutant_category_map": mappings,
            "source_ids": item.get("source_ids", []),
        })
    return {
        "well_id": request["well_id"],
        "overall_risk_models": package["overall_risk_models"],
        "risk_category_probabilities": categories,
        "candidate_enterprises": candidates,
        "category_score": category_score,
    }


def build_prompt(record: dict[str, Any]) -> str:
    payload = {k: v for k, v in record.items() if k != "category_score"}
    return (
        "<|im_start|>system\n/no_think\n"
        "你负责把井级机器学习风险类别证据与候选企业生产活动和空间信息进行约束关联。"
        "只在给定候选企业及其reported_pollutant_category_map中选择具体污染物。"
        "不得补充未列出的污染物，不得使用泛化类别名称代替具体污染物。<|im_end|>\n"
        "<|im_start|>user\n"
        "请从输入证据中排序10个候选污染物—企业组合。组合不得重复，企业可在具有多个明确污染物时重复。"
        "排序综合考虑两条光谱路径的风险类别概率、企业产品和原辅材料、调查记录的特征污染指标、"
        "企业运行状态、距离及区域水力关系。类别证据较弱时仍可保留候选，但排序应降低。"
        "只返回符合JSON schema的对象，不写解释性正文。\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        + "\n<|im_end|>\n<|im_start|>assistant\n/no_think\n"
    )


def allowed_pairs(record: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    allowed: dict[tuple[str, str], dict[str, Any]] = {}
    for candidate in record["candidate_enterprises"]:
        for pair in candidate["reported_pollutant_category_map"]:
            key = (candidate["enterprise_id"], str(pair["pollutant"]).casefold())
            allowed[key] = {
                "enterprise_id": candidate["enterprise_id"],
                "candidate_contaminant": pair["pollutant"],
                "risk_category": pair["risk_category"],
                "distance_rank": candidate.get("distance_rank") or 99,
            }
    return allowed


def normalize_links(raw_links: list[dict[str, Any]], record: dict[str, Any]) -> list[dict[str, Any]]:
    allowed = allowed_pairs(record)
    links: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in raw_links:
        key = (str(item.get("enterprise_id", "")), str(item.get("candidate_contaminant", "")).casefold())
        canonical = allowed.get(key)
        if not canonical or key in seen or item.get("risk_category") != canonical["risk_category"]:
            continue
        seen.add(key)
        links.append({
            "rank": len(links) + 1,
            "enterprise_id": canonical["enterprise_id"],
            "candidate_contaminant": canonical["candidate_contaminant"],
            "risk_category": canonical["risk_category"],
        })
    fallback = sorted(
        allowed.values(),
        key=lambda x: (-record["category_score"].get(x["risk_category"], 0.0), x["distance_rank"], x["enterprise_id"], x["candidate_contaminant"]),
    )
    for item in fallback:
        key = (item["enterprise_id"], str(item["candidate_contaminant"]).casefold())
        if key in seen:
            continue
        seen.add(key)
        links.append({
            "rank": len(links) + 1,
            "enterprise_id": item["enterprise_id"],
            "candidate_contaminant": item["candidate_contaminant"],
            "risk_category": item["risk_category"],
        })
        if len(links) == 10:
            break
    return links[:10]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--base-outputs", required=True)
    args = parser.parse_args()
    config = load_json(Path(args.config))
    request_path = Path(config["request_file"])
    output_dir = Path(config["output_dir"])
    requests = read_jsonl(request_path)
    base = {x["well_id"]: x for x in read_jsonl(Path(args.base_outputs))}
    method_lookup = load_method_lookup(Path(config["corpus_file"]))
    output_path = output_dir / "all_wells_qwen3_8b_contaminant_enterprise_links.jsonl"
    raw_path = output_dir / "all_wells_qwen3_8b_contaminant_enterprise_links_raw_responses.jsonl"
    log_path = output_dir / "all_wells_qwen3_8b_contaminant_enterprise_links_run_log.json"
    outputs: list[dict[str, Any]] = []
    raw_records: list[dict[str, Any]] = []
    logs: list[dict[str, Any]] = []
    compact_config = deepcopy(config)
    compact_config["max_output_tokens"] = 800
    compact_config["max_attempts"] = 1
    for index, request in enumerate(requests, start=1):
        well_id = request["well_id"]
        record = compact_record(request)
        started = time.perf_counter()
        response = post_completion(config["endpoint"], build_prompt(record), LINK_SCHEMA, compact_config)
        elapsed = time.perf_counter() - started
        raw_records.append({"well_id": well_id, "response": response})
        try:
            generated = json.loads(response["content"])
            generated_links = generated.get("links", [])
            parse_status = "parsed"
        except json.JSONDecodeError:
            generated_links = []
            parse_status = "fallback_after_parse_error"
        links = normalize_links(generated_links, record)
        merged = deepcopy(base[well_id])
        merged["contaminant_enterprise_links"] = links
        merged = normalize_deterministic_fields(merged, request, method_lookup)
        outputs.append(merged)
        logs.append({
            "well_id": well_id,
            "status": parse_status,
            "elapsed_seconds": round(elapsed, 3),
            "tokens_evaluated": response.get("tokens_evaluated"),
            "tokens_predicted": response.get("tokens_predicted"),
            "generated_valid_links": min(len(generated_links), 10),
            "final_links": len(links),
        })
        write_jsonl(raw_path, raw_records)
        write_jsonl(output_path, outputs)
        log_path.write_text(json.dumps({"items": logs}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[{index}/{len(requests)}] {well_id}: {parse_status}, {elapsed:.1f}s", flush=True)
    if len(outputs) != len(requests):
        raise RuntimeError("Not all wells were written")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
