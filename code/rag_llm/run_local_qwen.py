#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run reproducible local GroundwaterSpectra-LLM inference with llama.cpp.

The script reads only the provider-neutral inference request file. Independent
chemical evaluation answers are deliberately never opened.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator


REPRESENTATIVE_WELLS = ["C02", "A12", "B03", "C07"]

INDICATOR_ALIASES = {
    "pH": "pH", "CODMn": "耗氧量", "Ammonia-N": "氨氮", "Nitrite-N": "亚硝酸盐",
    "Nitrate-N": "硝酸盐", "Total hardness": "总硬度", "TDS": "溶解性总固体",
    "Sulfate": "硫酸盐", "Chloride": "氯化物", "Na": "钠", "Fluoride": "氟化物",
    "Iodide": "碘化物", "Sulfide": "硫化物", "Fe": "铁", "Mn": "锰", "Al": "铝",
    "Cu": "铜", "Zn": "锌", "Hg": "汞", "As": "砷", "Se": "硒", "Cd": "镉",
    "Cr(VI)": "六价铬", "Pb": "铅", "B": "硼", "Ni": "镍", "Co": "钴",
    "Volatile phenols": "挥发性酚类", "Anionic surfactants": "阴离子表面活性剂",
    "Dichloromethane": "二氯甲烷", "1,2-Dichloroethane": "1,2-二氯乙烷",
    "Vinyl chloride": "氯乙烯", "Chlorobenzene": "氯苯",
    "o-Dichlorobenzene": "1,2-二氯苯", "Naphthalene": "萘",
    "Benzo[a]pyrene": "苯并[a]芘", "Benzene": "苯", "Toluene": "甲苯",
    "Chloroform": "三氯甲烷", "Carbon tetrachloride": "四氯化碳", "Xylene": "二甲苯",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def load_method_lookup(corpus_path: Path) -> dict[str, dict[str, Any]]:
    records = read_jsonl(corpus_path)
    by_subject = {
        str(item["subject"]).strip(): item
        for item in records
        if item.get("knowledge_type") == "标准分析方法"
    }
    lookup: dict[str, dict[str, Any]] = {}
    for indicator_id, subject in INDICATOR_ALIASES.items():
        if subject in by_subject:
            record = dict(by_subject[subject])
            record["indicator_id"] = indicator_id
            record["source_id_list"] = [x for x in str(record.get("source_ids", "")).split(";") if x]
            record["chunk_id"] = f"{record['entry_id']}-C01"
            lookup[indicator_id.casefold()] = record
            lookup[subject.casefold()] = record
    return lookup


def build_prompt(request_record: dict[str, Any]) -> str:
    shared = {
        "category_model_performance": request_record["shared_category_model_performance"],
        "output_schema": request_record["output_schema"],
    }
    system_text = request_record["system_instruction"].rstrip()
    task_text = request_record["task_instruction"].rstrip()
    execution_rules = """

执行约束：
1. priority_risk_categories必须包含五类且各出现一次，rank为1至5且不重复；概率逐项复制输入值，不得改写。
2. supporting_evidence、counter_evidence和evidence_trace返回空数组，由程序根据输入原值生成证据编号和追溯记录。
3. confirmatory_tests先给出2至3项核心项目，最终扩展为10项分级筛查清单；所有项目必须具有适用分析方法，并引用supporting_evidence编号。
4. suspected_enterprises必须从当前井candidate_enterprises候选池中选择5家企业并排序为1至5。排序必须同时比较井企空间与区域水力关系、企业生产状态、行业、产品、原辅材料、特征污染指标及其与风险类别证据的一致性，不得直接沿用距离序位。
5. contaminant_enterprise_links必须输出10个不重复组合。企业限于当前候选池，candidate_contaminant必须逐字选自相应企业production_alignment.matched_reported_pollutants，risk_category必须与reported_pollutant_category_map一致。优先选择与井级类别概率、企业生产活动和空间关系共同一致的组合。
6. 不得补写标准方法、SHAP数值、企业距离、水力关系、污染物或来源编号；确定性字段由程序从知识库和结构化输入原值回填。
7. 每个文本字段限用一句简洁陈述，不复述完整输入。仅生成JSON对象。
""".strip()
    user_payload = {
        "task": task_text,
        "well_evidence_package": request_record["well_evidence_package"],
    }
    # Shared content precedes the variable well package so llama.cpp can reuse
    # the prompt prefix across the 41 sequential well-level requests.
    return (
        "<|im_start|>system\n"
        + system_text
        + "\n\n"
        + execution_rules
        + "\n\n以下为所有监测井共用的类别模型性能元数据和输出Schema：\n"
        + json.dumps(shared, ensure_ascii=False, separators=(",", ":"))
        + "<|im_end|>\n<|im_start|>user\n"
        + json.dumps(user_payload, ensure_ascii=False, separators=(",", ":"))
        + "\n/no_think<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def bounded_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Apply concise runtime bounds without changing the published field schema."""
    result = copy.deepcopy(schema)
    props = result["properties"]
    props["supporting_evidence"].update({"minItems": 4, "maxItems": 4})
    props["counter_evidence"].update({"maxItems": 1})
    props["priority_risk_categories"].update({"minItems": 5, "maxItems": 5})
    props["confirmatory_tests"].update({"minItems": 10, "maxItems": 10})
    props["contaminant_enterprise_links"].update({"minItems": 10, "maxItems": 10})
    props["contaminant_enterprise_links"]["items"]["properties"]["rank"]["maximum"] = 10
    props["suspected_enterprises"].update({"minItems": 5, "maxItems": 5})
    props["quality_control_checks"].update({"minItems": 1, "maxItems": 2})
    props["unexcluded_risks"].update({"minItems": 1, "maxItems": 2})
    props["uncertainty"].update({"minItems": 1, "maxItems": 3})
    props["evidence_trace"].update({"minItems": 2, "maxItems": 4})
    props["confirmatory_tests"]["items"]["properties"]["supporting_evidence_ids"]["maxItems"] = 3
    props["suspected_enterprises"]["items"]["properties"]["supporting_source_ids"]["maxItems"] = 4
    props["contaminant_enterprise_links"]["items"]["properties"]["supporting_source_ids"]["maxItems"] = 4
    trace_props = props["evidence_trace"]["items"]["properties"]
    trace_props["evidence_ids"]["maxItems"] = 4
    trace_props["feature_ids"]["maxItems"] = 4
    trace_props["rag_chunk_ids"]["maxItems"] = 10
    trace_props["source_ids"]["maxItems"] = 4
    result["$defs"]["evidence"]["properties"]["source_ids"]["maxItems"] = 4
    return result


def generation_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Constrain fields that are assembled deterministically after generation."""
    result = bounded_schema(schema)
    for name in ("supporting_evidence", "counter_evidence", "evidence_trace"):
        result["properties"][name].pop("minItems", None)
        result["properties"][name]["maxItems"] = 0
    result["properties"]["confirmatory_tests"].update({"minItems": 2, "maxItems": 3})
    return result


def post_completion(endpoint: str, prompt: str, schema: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "prompt": prompt,
        "json_schema": schema,
        "temperature": config["temperature"],
        "top_p": config["top_p"],
        "top_k": config["top_k"],
        "min_p": config["min_p"],
        "seed": config["seed"],
        "n_predict": config["max_output_tokens"],
        "stop": ["<|im_end|>", "<|endoftext|>"],
        "cache_prompt": True,
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    http_request = urllib.request.Request(
        endpoint.rstrip("/") + "/completion",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(http_request, timeout=config["timeout_seconds"]) as response:
        return json.load(response)


def collect_allowed(record: dict[str, Any], method_lookup: dict[str, dict[str, Any]] | None = None) -> dict[str, set[str]]:
    package = record["well_evidence_package"]
    feature_ids = set(package["spectral_features"]["eem"]) | set(package["spectral_features"]["uvvis"])
    chunks = package.get("rag_retrieval", {}).get("retrieved_chunks", [])
    chunk_ids = {c.get("chunk_id") for c in chunks if c.get("chunk_id")}
    source_ids = {s for c in chunks for s in c.get("source_ids", [])}
    method_subjects = {
        str(c.get("subject", "")).strip()
        for c in chunks
        if c.get("retrieval_domain") == "analytical_method" and c.get("subject")
    }
    candidates = package.get("enterprise_context", {}).get("candidate_enterprises", [])
    enterprise_ids = {
        str(c.get("enterprise_id") or c.get("internal_enterprise_id") or c.get("id"))
        for c in candidates
        if c.get("enterprise_id") or c.get("internal_enterprise_id") or c.get("id")
    }
    if method_lookup:
        chunk_ids.update(x["chunk_id"] for x in method_lookup.values())
        source_ids.update(s for x in method_lookup.values() for s in x["source_id_list"])
    return {
        "feature_ids": feature_ids,
        "chunk_ids": chunk_ids,
        "source_ids": source_ids,
        "method_subjects": method_subjects,
        "enterprise_ids": enterprise_ids,
    }


def normalize_deterministic_fields(
    output: dict[str, Any],
    request_record: dict[str, Any],
    method_lookup: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Copy non-generative identifiers and probabilities directly from model outputs."""
    package = request_record["well_evidence_package"]
    output["well_id"] = request_record["well_id"]
    output["model_status"] = package["overall_risk_models"]["model_status"]
    output["risk_screening_level"] = {
        "Both-high": "High Risk",
        "EEM-only high": "Risk Alert",
        "UV-Vis-only high": "Risk Alert",
        "Both-low": "Low Risk",
    }[output["model_status"]]
    expected = {
        item["category_name_zh"]: (float(item["eem"]["probability"]), float(item["uvvis"]["probability"]))
        for item in package["risk_category_models"]
    }
    for item in output.get("priority_risk_categories", []):
        category = item.get("category")
        if category in expected:
            item["eem_probability"], item["uvvis_probability"] = expected[category]
    evidence = []
    for modality, key in (("eem", "eem"), ("uv_vis", "uvvis")):
        block = package["overall_risk_models"][key]
        evidence.append({
            "evidence_id": f"EV-{len(evidence)+1:03d}", "evidence_type": "model", "modality": modality,
            "feature_id": None, "feature_family": None, "value_or_probability": float(block["probability"]),
            "shap_value": None, "source_ids": [],
            "statement": f"{modality}总体风险模型的折外概率为{float(block['probability']):.6f}。",
        })
    for modality, key in (("eem", "eem"), ("uv_vis", "uvvis")):
        block = package["overall_risk_models"][key]
        positive = block.get("top_positive_shap") or []
        negative = block.get("top_negative_shap") or []
        shap_item = (positive or negative or [None])[0]
        if shap_item:
            direction = "正向" if positive else "负向"
            evidence.append({
                "evidence_id": f"EV-{len(evidence)+1:03d}", "evidence_type": "model", "modality": modality,
                "feature_id": shap_item["feature_id"], "feature_family": shap_item["feature_family"],
                "value_or_probability": float(shap_item["raw_value"]), "shap_value": float(shap_item["shap_value"]),
                "source_ids": [],
                "statement": f"{shap_item['feature_id']}为{modality}总体风险模型中主要{direction}贡献特征之一；该贡献仅解释模型输出。",
            })
    output["supporting_evidence"] = evidence[:4]
    negative_candidates = []
    for modality, key in (("eem", "eem"), ("uv_vis", "uvvis")):
        for item in package["overall_risk_models"][key].get("top_negative_shap", []):
            negative_candidates.append((abs(float(item["shap_value"])), modality, item))
    output["counter_evidence"] = []
    if negative_candidates:
        _, modality, item = max(negative_candidates, key=lambda x: x[0])
        output["counter_evidence"] = [{
            "evidence_id": "EV-005", "evidence_type": "model", "modality": modality,
            "feature_id": item["feature_id"], "feature_family": item["feature_family"],
            "value_or_probability": float(item["raw_value"]), "shap_value": float(item["shap_value"]),
            "source_ids": [],
            "statement": f"{item['feature_id']}对{modality}总体风险概率产生负向贡献；该贡献不代表污染物浓度降低。",
        }]

    if method_lookup is not None:
        normalized_tests = []
        seen = set()
        category_by_indicator = {}
        for definition in request_record["shared_category_model_performance"]["category_definitions"]:
            for indicator in definition["included_indicators"]:
                category_by_indicator[indicator] = definition["category_name_zh"]
        for test in output.get("confirmatory_tests", []):
            method = method_lookup.get(str(test.get("indicator", "")).casefold())
            if not method or method["indicator_id"] in seen:
                continue
            indicator_id = method["indicator_id"]
            seen.add(indicator_id)
            normalized_tests.append({
                "indicator": indicator_id,
                "priority": len(normalized_tests) + 1,
                "purpose": str(test.get("purpose") or f"核查{category_by_indicator.get(indicator_id, '候选异常方向')}").strip(),
                "supporting_evidence_ids": ["EV-001", "EV-002"],
                "method_or_qc_note": f"[{method['entry_id']}; {';'.join(method['source_id_list'])}] {method['content']} {method['limitations']}",
                "_method": method,
            })
        if len(normalized_tests) < 10:
            ranked_categories = [x["category"] for x in sorted(output.get("priority_risk_categories", []), key=lambda x: x["rank"])]
            definitions = request_record["shared_category_model_performance"]["category_definitions"]
            for category in ranked_categories:
                definition = next((x for x in definitions if x["category_name_zh"] == category), None)
                if not definition:
                    continue
                for indicator_id in definition["included_indicators"]:
                    method = method_lookup.get(indicator_id.casefold())
                    if not method or indicator_id in seen:
                        continue
                    seen.add(indicator_id)
                    normalized_tests.append({
                        "indicator": indicator_id, "priority": len(normalized_tests) + 1,
                        "purpose": f"核查{category}", "supporting_evidence_ids": ["EV-001", "EV-002"],
                        "method_or_qc_note": f"[{method['entry_id']}; {';'.join(method['source_id_list'])}] {method['content']} {method['limitations']}",
                        "_method": method,
                    })
                    if len(normalized_tests) >= 10:
                        break
                if len(normalized_tests) >= 10:
                    break
        output["confirmatory_tests"] = normalized_tests[:10]

    candidate_map = {
        str(x.get("enterprise_id")): x
        for x in package.get("enterprise_context", {}).get("candidate_enterprises", [])
        if x.get("enterprise_id")
    }
    normalized_enterprises = []
    for item in output.get("suspected_enterprises", []):
        enterprise_id = str(item.get("enterprise_id", ""))
        candidate = candidate_map.get(enterprise_id)
        if not candidate or enterprise_id in {x["enterprise_id"] for x in normalized_enterprises}:
            continue
        relation = "；".join(filter(None, [
            f"距企业边界{candidate.get('distance_band')}" if candidate.get("distance_band") else "",
            f"位于井的{candidate.get('enterprise_relative_direction')}侧" if candidate.get("enterprise_relative_direction") else "",
            str(candidate.get("regional_hydraulic_consistency", "")),
            f"行业为{candidate.get('industry')}" if candidate.get("industry") else "",
            f"生产状态为{candidate.get('operating_status')}" if candidate.get("operating_status") else "",
            f"产品为{candidate.get('products')}" if candidate.get("products") and candidate.get("products") != "/" else "",
            f"原辅材料包括{candidate.get('raw_materials')}" if candidate.get("raw_materials") and candidate.get("raw_materials") != "/" else "",
            f"调查记录的特征污染指标为{candidate.get('characteristic_pollutants')}" if candidate.get("characteristic_pollutants") and candidate.get("characteristic_pollutants") != "/" else "",
            "与井级风险类别存在明确对应" if candidate.get("production_alignment", {}).get("model_category_alignment") else "",
        ]))
        normalized_enterprises.append({
            "enterprise_id": enterprise_id, "rank": len(normalized_enterprises) + 1,
            "relation_evidence": relation,
            "supporting_source_ids": list(candidate.get("source_ids", [])),
            "uncertainty": str(candidate.get("limitations", "")),
        })
    selected_ids = {x["enterprise_id"] for x in normalized_enterprises}
    for enterprise_id, candidate in candidate_map.items():
        if enterprise_id in selected_ids:
            continue
        relation = "；".join(filter(None, [
            f"距企业边界{candidate.get('distance_band')}" if candidate.get("distance_band") else "",
            f"位于井的{candidate.get('enterprise_relative_direction')}侧" if candidate.get("enterprise_relative_direction") else "",
            str(candidate.get("regional_hydraulic_consistency", "")),
            f"行业为{candidate.get('industry')}" if candidate.get("industry") else "",
            f"生产状态为{candidate.get('operating_status')}" if candidate.get("operating_status") else "",
            f"产品为{candidate.get('products')}" if candidate.get("products") and candidate.get("products") != "/" else "",
            f"原辅材料包括{candidate.get('raw_materials')}" if candidate.get("raw_materials") and candidate.get("raw_materials") != "/" else "",
            f"调查记录的特征污染指标为{candidate.get('characteristic_pollutants')}" if candidate.get("characteristic_pollutants") and candidate.get("characteristic_pollutants") != "/" else "",
            "与井级风险类别存在明确对应" if candidate.get("production_alignment", {}).get("model_category_alignment") else "",
        ]))
        normalized_enterprises.append({
            "enterprise_id": enterprise_id, "rank": len(normalized_enterprises) + 1,
            "relation_evidence": relation,
            "supporting_source_ids": list(candidate.get("source_ids", [])),
            "uncertainty": str(candidate.get("limitations", "")),
        })
        if len(normalized_enterprises) >= 5:
            break
    for rank, item in enumerate(normalized_enterprises[:5], start=1):
        item["rank"] = rank
    output["suspected_enterprises"] = normalized_enterprises[:5]

    # Candidate contaminant-enterprise links are restricted to explicit,
    # enterprise-reported pollutants that map to one of the five modelled
    # chemical-risk categories. Generated text may rank the links, while IDs,
    # pollutant names, categories and source IDs are copied from frozen input.
    allowed_pairs: dict[tuple[str, str], dict[str, str]] = {}
    for enterprise_id, candidate in candidate_map.items():
        for pair in candidate.get("production_alignment", {}).get("reported_pollutant_category_map", []):
            pollutant = str(pair.get("pollutant", "")).strip()
            category = str(pair.get("risk_category", "")).strip()
            if pollutant and category:
                allowed_pairs[(enterprise_id, pollutant.casefold())] = {
                    "pollutant": pollutant, "risk_category": category,
                }
    normalized_links = []
    used_pairs: set[tuple[str, str]] = set()
    for item in output.get("contaminant_enterprise_links", []):
        enterprise_id = str(item.get("enterprise_id", "")).strip()
        contaminant_key = str(item.get("candidate_contaminant", "")).strip().casefold()
        canonical = allowed_pairs.get((enterprise_id, contaminant_key))
        candidate = candidate_map.get(enterprise_id)
        pair_key = (enterprise_id, canonical["pollutant"].casefold()) if canonical else None
        if not canonical or not candidate or pair_key in used_pairs:
            continue
        used_pairs.add(pair_key)
        normalized_links.append({
            "rank": len(normalized_links) + 1,
            "enterprise_id": enterprise_id,
            "candidate_contaminant": canonical["pollutant"],
            "risk_category": canonical["risk_category"],
            "linkage_evidence": str(item.get("linkage_evidence") or f"{canonical['pollutant']}为{enterprise_id}调查记录的特征污染指标，并与井级{canonical['risk_category']}证据对应。").strip(),
            "supporting_source_ids": list(candidate.get("source_ids", [])),
            "uncertainty": str(candidate.get("limitations", "")),
        })
    category_rank = {
        item["category"]: int(item["rank"])
        for item in output.get("priority_risk_categories", [])
        if item.get("category") and item.get("rank")
    }
    fallback_pairs = []
    for enterprise_id, candidate in candidate_map.items():
        for pair in candidate.get("production_alignment", {}).get("reported_pollutant_category_map", []):
            pollutant = str(pair.get("pollutant", "")).strip()
            category = str(pair.get("risk_category", "")).strip()
            if pollutant and category:
                fallback_pairs.append((
                    category_rank.get(category, 99), int(candidate.get("distance_rank", 99)),
                    enterprise_id, pollutant, category, candidate,
                ))
    for _, _, enterprise_id, pollutant, category, candidate in sorted(fallback_pairs):
        pair_key = (enterprise_id, pollutant.casefold())
        if pair_key in used_pairs:
            continue
        used_pairs.add(pair_key)
        normalized_links.append({
            "rank": len(normalized_links) + 1,
            "enterprise_id": enterprise_id,
            "candidate_contaminant": pollutant,
            "risk_category": category,
            "linkage_evidence": f"{pollutant}为{enterprise_id}调查记录的特征污染指标，并与井级{category}证据对应。",
            "supporting_source_ids": list(candidate.get("source_ids", [])),
            "uncertainty": str(candidate.get("limitations", "")),
        })
        if len(normalized_links) >= 10:
            break
    output["contaminant_enterprise_links"] = normalized_links[:10]
    for rank, item in enumerate(output["contaminant_enterprise_links"], start=1):
        item["rank"] = rank

    spectral_chunks = [x for x in package.get("rag_retrieval", {}).get("retrieved_chunks", []) if x.get("retrieval_domain") == "spectral_interpretation"]
    contribution_ids = ["EV-003", "EV-004"] + (["EV-005"] if output["counter_evidence"] else [])
    traces = [{
        "claim_id": "overall-risk", "evidence_ids": ["EV-001", "EV-002"], "feature_ids": [],
        "rag_chunk_ids": [], "source_ids": [],
    }, {
        "claim_id": "spectral-contribution", "evidence_ids": contribution_ids,
        "feature_ids": [x["feature_id"] for x in output["supporting_evidence"][2:]] + [x["feature_id"] for x in output["counter_evidence"]],
        "rag_chunk_ids": [x["chunk_id"] for x in spectral_chunks[:3]],
        "source_ids": list(dict.fromkeys(s for x in spectral_chunks[:3] for s in x.get("source_ids", [])))[:4],
    }]
    tests = output.get("confirmatory_tests", [])
    if tests:
        methods = [x.pop("_method") for x in tests]
        traces.append({
            "claim_id": "confirmatory-tests", "evidence_ids": ["EV-001", "EV-002"], "feature_ids": [],
            "rag_chunk_ids": [x["chunk_id"] for x in methods],
            "source_ids": list(dict.fromkeys(s for x in methods for s in x["source_id_list"]))[:4],
        })
    if output["suspected_enterprises"] or output.get("contaminant_enterprise_links"):
        site_chunks = [x for x in package.get("rag_retrieval", {}).get("retrieved_chunks", []) if x.get("retrieval_domain") == "site_or_enterprise"]
        source_items = output.get("contaminant_enterprise_links") or output["suspected_enterprises"]
        traces.append({
            "claim_id": "contaminant-enterprise-links", "evidence_ids": ["EV-001", "EV-002"], "feature_ids": [],
            "rag_chunk_ids": [x["chunk_id"] for x in site_chunks[:3]],
            "source_ids": list(dict.fromkeys(s for x in source_items for s in x["supporting_source_ids"]))[:4],
        })
    output["evidence_trace"] = traces[:4]
    return output


def semantic_errors(output: dict[str, Any], request_record: dict[str, Any], method_lookup: dict[str, dict[str, Any]] | None = None) -> list[str]:
    errors: list[str] = []
    package = request_record["well_evidence_package"]
    allowed = collect_allowed(request_record, method_lookup)
    if output.get("well_id") != request_record["well_id"]:
        errors.append("well_id与输入不一致")
    expected_status = package["overall_risk_models"]["model_status"]
    if output.get("model_status") != expected_status:
        errors.append("model_status与输入不一致")

    expected_categories = {
        item["category_name_zh"]: (item["eem"]["probability"], item["uvvis"]["probability"])
        for item in package["risk_category_models"]
    }
    categories = output.get("priority_risk_categories", [])
    if len(categories) != 5 or {x.get("category") for x in categories} != set(expected_categories):
        errors.append("priority_risk_categories未完整且唯一地包含五类")
    if {x.get("rank") for x in categories} != {1, 2, 3, 4, 5}:
        errors.append("五类风险rank必须为不重复的1至5")
    for item in categories:
        category = item.get("category")
        if category in expected_categories:
            eem, uvvis = expected_categories[category]
            if abs(float(item.get("eem_probability", -9)) - float(eem)) > 1e-6:
                errors.append(f"{category}的EEM概率未原值复制")
            if abs(float(item.get("uvvis_probability", -9)) - float(uvvis)) > 1e-6:
                errors.append(f"{category}的UV-Vis概率未原值复制")

    supporting = output.get("supporting_evidence", [])
    counter = output.get("counter_evidence", [])
    evidence = supporting + counter
    evidence_ids = [x.get("evidence_id") for x in evidence]
    if len(evidence_ids) != len(set(evidence_ids)):
        errors.append("evidence_id存在重复")
    expected_ids = [f"EV-{i:03d}" for i in range(1, len(evidence_ids) + 1)]
    if evidence_ids != expected_ids:
        errors.append("evidence_id未按EV-001起连续编号")
    supporting_ids = {x.get("evidence_id") for x in supporting}
    all_ids = set(evidence_ids)
    for item in evidence:
        fid = item.get("feature_id")
        if fid is not None and fid not in allowed["feature_ids"]:
            errors.append(f"证据引用未知feature_id: {fid}")
        for sid in item.get("source_ids", []):
            if sid not in allowed["source_ids"]:
                errors.append(f"证据引用未知source_id: {sid}")
    for test in output.get("confirmatory_tests", []):
        if method_lookup is not None and str(test.get("indicator", "")).casefold() not in method_lookup:
            errors.append(f"检测项目没有知识库标准方法: {test.get('indicator')}")
        for ref in test.get("supporting_evidence_ids", []):
            if ref not in supporting_ids:
                errors.append(f"检测项目引用未定义的支持证据: {ref}")
    for trace in output.get("evidence_trace", []):
        for ref in trace.get("evidence_ids", []):
            if ref not in all_ids:
                errors.append(f"证据链引用未定义证据: {ref}")
        for fid in trace.get("feature_ids", []):
            if fid not in allowed["feature_ids"]:
                errors.append(f"证据链引用未知feature_id: {fid}")
        for cid in trace.get("rag_chunk_ids", []):
            if cid not in allowed["chunk_ids"]:
                errors.append(f"证据链引用未知rag_chunk_id: {cid}")
        for sid in trace.get("source_ids", []):
            if sid not in allowed["source_ids"]:
                errors.append(f"证据链引用未知source_id: {sid}")
    links = output.get("contaminant_enterprise_links", [])
    if len(links) != 10 or len({(x.get("enterprise_id"), x.get("candidate_contaminant")) for x in links}) != 10:
        errors.append("contaminant_enterprise_links必须包含10个不重复候选组合")
    if {x.get("rank") for x in links} != set(range(1, 11)):
        errors.append("候选污染物—企业组合rank必须为不重复的1至10")
    candidate_map = {
        str(x.get("enterprise_id")): x
        for x in package.get("enterprise_context", {}).get("candidate_enterprises", [])
        if x.get("enterprise_id")
    }
    for link in links:
        enterprise_id = link.get("enterprise_id")
        candidate = candidate_map.get(enterprise_id)
        allowed_pairs = {
            (str(x.get("pollutant", "")).strip(), str(x.get("risk_category", "")).strip())
            for x in (candidate or {}).get("production_alignment", {}).get("reported_pollutant_category_map", [])
        }
        if candidate is None:
            errors.append(f"污染物—企业组合引用当前候选池之外企业: {enterprise_id}")
        if (link.get("candidate_contaminant"), link.get("risk_category")) not in allowed_pairs:
            errors.append(f"污染物—企业组合缺少企业特征污染指标映射: {enterprise_id}/{link.get('candidate_contaminant')}")
        for sid in link.get("supporting_source_ids", []):
            if sid not in allowed["source_ids"]:
                errors.append(f"污染物—企业组合引用未知source_id: {sid}")

    enterprises = output.get("suspected_enterprises", [])
    if len(enterprises) != 5 or len({x.get("enterprise_id") for x in enterprises}) != 5:
        errors.append("suspected_enterprises必须包含5家不重复候选企业")
    if {x.get("rank") for x in enterprises} != {1, 2, 3, 4, 5}:
        errors.append("候选企业rank必须为不重复的1至5")
    for enterprise in enterprises:
        if enterprise.get("enterprise_id") not in allowed["enterprise_ids"]:
            errors.append(f"疑似企业不在当前井候选清单: {enterprise.get('enterprise_id')}")
        for sid in enterprise.get("supporting_source_ids", []):
            if sid not in allowed["source_ids"]:
                errors.append(f"疑似企业引用未知source_id: {sid}")
    return sorted(set(errors))


def schema_errors(output: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    validator = Draft202012Validator(schema)
    return [f"JSON Schema: {'/'.join(map(str, err.path))}: {err.message}" for err in validator.iter_errors(output)]


def repair_prompt(original_prompt: str, invalid_output: dict[str, Any], errors: list[str]) -> str:
    suffix = (
        "<|im_start|>user\n上一版JSON未通过一致性检查。请只修正下列错误，保持所有输入事实和概率不变，并重新输出完整JSON：\n"
        + json.dumps(errors, ensure_ascii=False)
        + "\n上一版JSON：\n"
        + json.dumps(invalid_output, ensure_ascii=False, separators=(",", ":"))
        + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    # Remove the final assistant prefill, then add the repair turn.
    marker = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    return original_prompt[: -len(marker)] + marker + json.dumps(invalid_output, ensure_ascii=False) + "<|im_end|>\n" + suffix


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=["representative", "all"], default="representative")
    parser.add_argument("--wells", nargs="*")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional output directory. Relative paths are resolved from the repository root.",
    )
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    project_root = Path(__file__).resolve().parents[2]
    for key in ("request_file", "corpus_file", "output_dir", "model_file"):
        candidate = Path(config[key])
        if not candidate.is_absolute():
            config[key] = str(project_root / candidate)
    if args.output_dir is not None:
        output_override = args.output_dir
        if not output_override.is_absolute():
            output_override = project_root / output_override
        config["output_dir"] = str(output_override)
    method_lookup = load_method_lookup(Path(config["corpus_file"]))
    input_path = Path(config["request_file"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    requests = read_jsonl(input_path)
    requested = args.wells or (REPRESENTATIVE_WELLS if args.mode == "representative" else [r["well_id"] for r in requests])
    records = [r for r in requests if r["well_id"] in requested]
    if {r["well_id"] for r in records} != set(requested):
        raise ValueError("Requested well IDs are not all present in the request file")

    prefix = "representative_wells" if args.mode == "representative" else "all_wells"
    output_path = output_dir / f"{prefix}_qwen3_8b_outputs.jsonl"
    raw_path = output_dir / f"{prefix}_qwen3_8b_raw_responses.jsonl"
    log_path = output_dir / f"{prefix}_qwen3_8b_run_log.json"
    finished: list[dict[str, Any]] = read_jsonl(output_path) if args.resume and output_path.exists() else []
    raw_records: list[dict[str, Any]] = read_jsonl(raw_path) if args.resume and raw_path.exists() else []
    run_items: list[dict[str, Any]] = []
    if args.resume and log_path.exists():
        run_items = json.loads(log_path.read_text(encoding="utf-8-sig")).get("items", [])
    completed_ids = {item["well_id"] for item in finished}
    records = [record for record in records if record["well_id"] not in completed_ids]

    for index, record in enumerate(records, start=1):
        well_id = record["well_id"]
        published_schema = record["output_schema"]
        schema = generation_schema(published_schema)
        prompt = build_prompt(record)
        attempt_logs = []
        result = None
        errors: list[str] = []
        for attempt in range(1, config["max_attempts"] + 1):
            started = time.perf_counter()
            response = post_completion(config["endpoint"], prompt, schema, config)
            elapsed = time.perf_counter() - started
            content = response["content"].strip()
            try:
                result = normalize_deterministic_fields(json.loads(content), record, method_lookup)
                errors = schema_errors(result, bounded_schema(published_schema)) + semantic_errors(result, record, method_lookup)
            except json.JSONDecodeError as exc:
                result = None
                errors = [f"JSON解析失败: {exc}"]
            attempt_logs.append({
                "attempt": attempt,
                "elapsed_seconds": round(elapsed, 3),
                "tokens_evaluated": response.get("tokens_evaluated"),
                "tokens_predicted": response.get("tokens_predicted"),
                "timings": response.get("timings", {}),
                "validation_errors": errors,
            })
            raw_records.append({"well_id": well_id, "attempt": attempt, "response": response})
            write_jsonl(raw_path, raw_records)
            log_path.write_text(json.dumps({"items": run_items, "current_well": well_id, "current_attempts": attempt_logs}, ensure_ascii=False, indent=2), encoding="utf-8")
            if result is not None and not errors:
                break
            if result is not None and attempt < config["max_attempts"]:
                prompt = repair_prompt(prompt, result, errors)
        if result is None or errors:
            run_items.append({"well_id": well_id, "status": "failed", "attempts": attempt_logs, "errors": errors})
            write_jsonl(raw_path, raw_records)
            log_path.write_text(json.dumps({"items": run_items}, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[{index}/{len(records)}] {well_id}: FAILED: {'; '.join(errors)}", flush=True)
            continue
        finished.append(result)
        run_items.append({"well_id": well_id, "status": "passed", "attempts": attempt_logs})
        write_jsonl(output_path, finished)
        write_jsonl(raw_path, raw_records)
        log_path.write_text(json.dumps({"items": run_items}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[{index}/{len(records)}] {well_id}: PASSED in {sum(x['elapsed_seconds'] for x in attempt_logs):.1f}s", flush=True)
        print(f"\n===== LLM OUTPUT: {well_id} =====", flush=True)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_repository": config["model_repository"],
        "model_revision": config["model_revision"],
        "model_file": config["model_file"],
        "model_sha256": config["model_sha256"],
        "llama_cpp_version": config["llama_cpp_version"],
        "backend": config["backend"],
        "context_size": config["context_size"],
        "inference_parameters": {k: config[k] for k in ["temperature", "top_p", "top_k", "min_p", "seed", "max_output_tokens"]},
        "request_file": str(input_path),
        "request_file_sha256": sha256(input_path),
        "mode": args.mode,
        "requested_wells": requested,
        "passed_wells": [x["well_id"] for x in run_items if x["status"] == "passed"],
        "failed_wells": [x["well_id"] for x in run_items if x["status"] == "failed"],
        "independent_evaluation_answers_loaded": False,
        "output_file": str(output_path),
        "raw_response_file": str(raw_path),
        "run_log": str(log_path),
    }
    (output_dir / f"{prefix}_qwen3_8b_reproducibility_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    passed_ids = {x["well_id"] for x in finished} & set(requested)
    failed_ids = set(requested) - passed_ids
    print(json.dumps({"passed": len(passed_ids), "failed": len(failed_ids), "output": str(output_path)}, ensure_ascii=False))
    return 0 if not failed_ids else 1


if __name__ == "__main__":
    raise SystemExit(main())
