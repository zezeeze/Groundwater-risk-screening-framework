#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build and audit the GroundwaterSpectra hybrid RAG retrieval layer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import random
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).with_name("rag_config.json")


def resolve_cached_model(model_id: str, markers: tuple[str, ...], revision: str | None = None) -> str:
    """Resolve a complete local HF snapshot and avoid partial-cache selection."""
    model_path = Path(str(model_id))
    if model_path.exists():
        return str(model_path)
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    if hub.name != "hub":
        hub = hub / "hub"
    snapshots = hub / ("models--" + str(model_id).replace("/", "--")) / "snapshots"
    if snapshots.exists():
        if revision:
            fixed_snapshot = snapshots / revision
            if fixed_snapshot.is_dir() and all((fixed_snapshot / marker).exists() for marker in markers):
                return str(fixed_snapshot)
            raise FileNotFoundError(
                f"Required Hugging Face snapshot is not cached: {model_id}@{revision}"
            )
        for snapshot in sorted(snapshots.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if snapshot.is_dir() and all((snapshot / marker).exists() for marker in markers):
                return str(snapshot)
    return str(model_id)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def resolve_paths(config: dict) -> dict[str, Path]:
    return {key: PROJECT_ROOT / value for key, value in config["paths"].items()}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def entry_text(entry: dict) -> str:
    source_ids = entry.get("source_ids", "")
    if isinstance(source_ids, list):
        source_ids = ";".join(source_ids)
    fields = [
        f"知识类型：{entry.get('knowledge_type', '')}",
        f"主题：{entry.get('subject', '')}",
        f"类别：{entry.get('category', '')}",
        f"内容：{entry.get('content', '')}",
        f"适用范围：{entry.get('scope', '')}",
        f"限制与边界：{entry.get('limitations', '')}",
        f"来源编号：{source_ids}",
    ]
    return "\n".join(fields)


def split_entry(entry: dict, tokenizer, chunk_size: int, overlap: int) -> list[dict]:
    text = entry_text(entry)
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) <= chunk_size:
        pieces = [text]
    else:
        pieces = []
        step = chunk_size - overlap
        for start in range(0, len(token_ids), step):
            part = token_ids[start : start + chunk_size]
            if not part:
                break
            pieces.append(tokenizer.decode(part, skip_special_tokens=True))
            if start + chunk_size >= len(token_ids):
                break

    source_ids = entry.get("source_ids", "")
    if isinstance(source_ids, str):
        source_ids = [x.strip() for x in source_ids.split(";") if x.strip()]
    elif source_ids is None:
        source_ids = []

    chunks = []
    for idx, piece in enumerate(pieces, start=1):
        chunks.append(
            {
                "chunk_id": f"{entry['entry_id']}-C{idx:02d}",
                "entry_id": entry["entry_id"],
                "chunk_index": idx,
                "chunk_count": len(pieces),
                "knowledge_type": entry.get("knowledge_type"),
                "subject": entry.get("subject"),
                "category": entry.get("category"),
                "content": piece,
                "scope": entry.get("scope"),
                "limitations": entry.get("limitations"),
                "source_ids": source_ids,
                "status": entry.get("status"),
                "updated": entry.get("updated"),
                "token_count": len(tokenizer.encode(piece, add_special_tokens=False)),
            }
        )
    return chunks


def bm25_tokens(text: str) -> list[str]:
    import jieba

    words = []
    for token in jieba.lcut(text.lower(), cut_all=False):
        token = token.strip()
        if not token:
            continue
        if re.fullmatch(r"[\W_]+", token):
            continue
        words.append(token)
    words.extend(re.findall(r"[a-z]+(?:[-_/][a-z0-9]+)*|\d+(?:\.\d+)?", text.lower()))
    return words


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(matrix, axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    return matrix / denom


def build(config: dict, paths: dict[str, Path]) -> None:
    import faiss
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer
    from rank_bm25 import BM25Okapi

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    random.seed(config["random_seed"])
    np.random.seed(config["random_seed"])

    embedding_path = resolve_cached_model(
        config["embedding_model"],
        ("modules.json", "config.json", "tokenizer.json"),
        config.get("embedding_revision"),
    )
    tokenizer = AutoTokenizer.from_pretrained(embedding_path, local_files_only=True)
    entries = read_jsonl(paths["corpus"])
    chunks = []
    for entry in entries:
        chunks.extend(
            split_entry(
                entry,
                tokenizer,
                config["chunk_size_tokens"],
                config["chunk_overlap_tokens"],
            )
        )

    model = SentenceTransformer(
        embedding_path,
        device="cpu", local_files_only=True
    )
    model.max_seq_length = config["max_sequence_length"]
    embeddings = model.encode(
        [item["content"] for item in chunks],
        batch_size=config["embedding_batch_size"],
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")
    embeddings = normalize_rows(embeddings).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    tokenized = [bm25_tokens(item["content"]) for item in chunks]
    bm25 = BM25Okapi(tokenized)

    index_dir = paths["index_dir"]
    index_dir.mkdir(parents=True, exist_ok=True)
    np.asarray(faiss.serialize_index(index), dtype=np.uint8).tofile(index_dir / "faiss.index")
    np.save(index_dir / "embeddings.npy", embeddings)
    write_jsonl(index_dir / "chunks.jsonl", chunks)
    with (index_dir / "bm25.pkl").open("wb") as handle:
        pickle.dump({"bm25": bm25, "tokenized_corpus": tokenized}, handle)

    manifest = {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "corpus_sha256": sha256(paths["corpus"]),
        "n_entries": len(entries),
        "n_chunks": len(chunks),
        "embedding_dimension": int(embeddings.shape[1]),
        "embedding_model": config["embedding_model"],
        "embedding_revision": config.get("embedding_revision"),
        "chunk_size_tokens": config["chunk_size_tokens"],
        "chunk_overlap_tokens": config["chunk_overlap_tokens"],
        "faiss_index": "IndexFlatIP",
        "similarity": "cosine similarity after L2 normalization",
        "bm25": "BM25Okapi",
        "files": {
            "faiss.index": sha256(index_dir / "faiss.index"),
            "embeddings.npy": sha256(index_dir / "embeddings.npy"),
            "chunks.jsonl": sha256(index_dir / "chunks.jsonl"),
            "bm25.pkl": sha256(index_dir / "bm25.pkl"),
        },
    }
    write_json(index_dir / "index_manifest.json", manifest)


def metadata_lookup(metadata: dict) -> dict[tuple[str, str], dict]:
    return {(row["spectral_path"], row["category_id"]): row for row in metadata["records"]}


def build_queries(
    well: dict,
    performance: dict[tuple[str, str], dict],
    category_definitions: dict[str, dict],
) -> tuple[dict[str, str], dict]:
    overall = well["overall_risk_models"]
    categories = []
    for item in well["risk_category_models"]:
        category_en = item["category_id"]
        category_zh = item["category_name_zh"]
        eem = float(item["eem"]["probability"])
        uv = float(item["uvvis"]["probability"])
        categories.append((max(eem, uv), category_en, category_zh, eem, uv))
    categories.sort(reverse=True)
    top_categories = categories[:3]

    feature_ids = list(well.get("rag_retrieval", {}).get("query_terms", {}).get("feature_ids", []))
    feature_mappings = {}
    for model_block in [overall["eem"], overall["uvvis"]]:
        for direction in ("top_positive_shap", "top_negative_shap"):
            for item in model_block.get(direction, []):
                feature_mappings[item["feature_id"]] = item.get("evidence_mapping", "")
                if item["feature_id"] not in feature_ids:
                    feature_ids.append(item["feature_id"])
    feature_ids = list(dict.fromkeys(feature_ids))
    feature_terms = [
        f"{feature_id} {feature_mappings.get(feature_id, '')}".strip()
        for feature_id in feature_ids
    ]

    category_lines = []
    indicator_terms = []
    for _, category_en, category_zh, eem, uv in top_categories:
        eem_perf = performance.get(("eem", category_en), {})
        uv_perf = performance.get(("uv_vis", category_en), {})
        definition = category_definitions.get(category_en, {})
        indicators = definition.get("included_indicators", [])
        indicator_terms.extend(indicators)
        category_lines.append(
            f"{category_zh} EEM概率{eem:.3f} UV-Vis概率{uv:.3f} "
            f"EEM PR-AUC{eem_perf.get('oof_pr_auc', 'NA')} "
            f"UV-Vis PR-AUC{uv_perf.get('oof_pr_auc', 'NA')} "
            f"包含指标{'、'.join(indicators)}"
        )
    indicator_terms = list(dict.fromkeys(indicator_terms))

    candidates = well.get("enterprise_context", {}).get("candidate_enterprises", [])
    candidate_lines = [
        f"{x.get('enterprise_id')} 距离{x.get('distance_band', '')} "
        f"方向{x.get('enterprise_relative_direction', '')} "
        f"水力关系{x.get('regional_hydraulic_consistency', '')}"
        for x in candidates
        if x.get("enterprise_id")
    ]
    site = well.get("site_context", {})

    queries = {
        "spectral_interpretation": "\n".join(
            [
                f"监测井{well['well_id']}，总体模型状态{overall['model_status']}。",
                "需要解释的EEM和UV-Vis特征：" + "；".join(feature_terms),
                "检索光谱特征的环境意义、响应边界、双光谱差异、混合探针限制、弱响应和模型分歧解释规则。",
            ]
        ),
        "chemical_direction": "\n".join(
            [
                "候选化学风险类别：" + "；".join(category_lines),
                "候选指标：" + "、".join(indicator_terms),
                "检索这些类别与指标的环境意义、地下水行为、风险标签映射和不能由光谱直接确定的边界。",
            ]
        ),
        "analytical_method": "\n".join(
            [
                "需要确认的风险类别：" + "；".join(x[2] for x in top_categories),
                "候选确认性检测指标：" + "、".join(indicator_terms),
                "检索相应的标准分析方法、适用范围、样品保存、前处理、稀释、基质效应和质量控制要求。",
            ]
        ),
        "site_or_enterprise": "\n".join(
            [
                f"监测井：{well['well_id']}；监测角色：{site.get('monitoring_role', '')}；"
                f"含水层或岩性：{site.get('aquifer_or_lithology', '')}；"
                f"企业边界关系：{site.get('enterprise_location_status', '')}。",
                "候选企业关系：" + "；".join(candidate_lines),
                "检索该井场地背景、水文地质、井企空间关系，以及候选企业产品、原辅材料和特征污染指标。"
                "空间邻近不能单独作为污染来源结论。",
            ]
        ),
    }

    query_meta = {
        "well_id": well["well_id"],
        "model_status": overall["model_status"],
        "eem_overall_probability": float(overall["eem"]["probability"]),
        "uvvis_overall_probability": float(overall["uvvis"]["probability"]),
        "top_risk_categories": [
            {
                "category_id": category_en,
                "category": category_zh,
                "eem_probability": eem,
                "uvvis_probability": uv,
            }
            for _, category_en, category_zh, eem, uv in top_categories
        ],
        "feature_ids": feature_ids,
        "candidate_enterprise_ids": [
            x.get("enterprise_id") for x in candidates if x.get("enterprise_id")
        ],
    }
    return queries, query_meta


def retrieve(config: dict, paths: dict[str, Path]) -> None:
    import faiss
    from sentence_transformers import CrossEncoder, SentenceTransformer

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    index_dir = paths["index_dir"]
    chunks = read_jsonl(index_dir / "chunks.jsonl")
    embeddings = np.load(index_dir / "embeddings.npy").astype("float32")
    with (index_dir / "bm25.pkl").open("rb") as handle:
        bm25_bundle = pickle.load(handle)
    bm25 = bm25_bundle["bm25"]

    domain_indices = {}
    domain_allowed_ids = {}
    for domain, settings in config["retrieval_domains"].items():
        allowed = np.array(
            [
                idx
                for idx, chunk in enumerate(chunks)
                if chunk.get("knowledge_type") in settings["knowledge_types"]
            ],
            dtype="int64",
        )
        subindex = faiss.IndexIDMap2(faiss.IndexFlatIP(embeddings.shape[1]))
        subindex.add_with_ids(embeddings[allowed], allowed)
        domain_indices[domain] = subindex
        domain_allowed_ids[domain] = allowed

    embedder = SentenceTransformer(
        resolve_cached_model(
            config["embedding_model"],
            ("modules.json", "config.json", "tokenizer.json"),
            config.get("embedding_revision"),
        ),
        device="cpu", local_files_only=True
    )
    embedder.max_seq_length = config["max_sequence_length"]
    reranker = CrossEncoder(
        resolve_cached_model(
            config["reranker_model"],
            ("config.json", "tokenizer.json"),
            config.get("reranker_revision"),
        ),
        device="cpu",
        local_files_only=True,
        max_length=config["max_sequence_length"],
    )

    wells = read_jsonl(paths["well_inputs"])
    shared = read_json(paths["shared_model_metadata"])
    performance = metadata_lookup(shared)
    category_definitions = {
        item["category_id"]: item for item in shared["category_definitions"]
    }

    outputs = []
    started = time.time()
    domain_order = list(config["retrieval_domains"])
    for order, well in enumerate(wells, start=1):
        queries, query_meta = build_queries(well, performance, category_definitions)
        query_embeddings = embedder.encode(
            [queries[domain] for domain in domain_order],
            batch_size=len(domain_order),
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        shortlists = {}
        pair_records = []
        for domain_index, domain in enumerate(domain_order):
            query = queries[domain]
            if domain == "site_or_enterprise":
                relevant_subjects = {well["well_id"], *query_meta["candidate_enterprise_ids"]}
                allowed = np.array(
                    [
                        idx
                        for idx in domain_allowed_ids[domain]
                        if chunks[int(idx)].get("subject") in relevant_subjects
                        or chunks[int(idx)].get("knowledge_type") in {"场地环境知识", "水文地质知识"}
                    ],
                    dtype="int64",
                )
                search_index = faiss.IndexIDMap2(faiss.IndexFlatIP(embeddings.shape[1]))
                search_index.add_with_ids(embeddings[allowed], allowed)
            else:
                allowed = domain_allowed_ids[domain]
                search_index = domain_indices[domain]
            top_k = min(config["vector_top_k"], len(allowed))
            vector_scores, vector_indices = search_index.search(
                query_embeddings[domain_index : domain_index + 1],
                top_k,
            )
            vector_rank = {
                int(idx): rank
                for rank, idx in enumerate(vector_indices[0], start=1)
                if int(idx) >= 0
            }

            bm25_scores = bm25.get_scores(bm25_tokens(query))
            bm25_allowed = allowed[np.argsort(-bm25_scores[allowed])[: min(config["bm25_top_k"], len(allowed))]]
            bm25_rank = {int(idx): rank for rank, idx in enumerate(bm25_allowed, start=1)}

            candidates = sorted(set(vector_rank) | set(bm25_rank))
            fused = []
            for idx in candidates:
                rrf = 0.0
                if idx in vector_rank:
                    rrf += 1.0 / (config["rrf_constant"] + vector_rank[idx])
                if idx in bm25_rank:
                    rrf += 1.0 / (config["rrf_constant"] + bm25_rank[idx])
                fused.append(
                    {
                        "idx": idx,
                        "rrf_score": rrf,
                        "vector_rank": vector_rank.get(idx),
                        "bm25_rank": bm25_rank.get(idx),
                    }
                )
            fused.sort(key=lambda item: item["rrf_score"], reverse=True)
            shortlist = []
            for required_type in config["retrieval_domains"][domain].get(
                "required_knowledge_types", []
            ):
                match = next(
                    (
                        item
                        for item in fused
                        if chunks[item["idx"]].get("knowledge_type") == required_type
                    ),
                    None,
                )
                if match is not None and match not in shortlist:
                    shortlist.append(match)
            for item in fused:
                if len(shortlist) >= config["rrf_shortlist_k_per_domain"]:
                    break
                if item not in shortlist:
                    shortlist.append(item)
            shortlists[domain] = shortlist
            for item in shortlist:
                pair_records.append((domain, item, (query, chunks[item["idx"]]["content"])))

        rerank_scores = reranker.predict(
            [item[2] for item in pair_records],
            batch_size=config["reranker_batch_size"],
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        scored_by_domain = {domain: [] for domain in domain_order}
        for (domain, item, _), score in zip(pair_records, rerank_scores):
            enriched = dict(item)
            enriched["reranker_score"] = float(score)
            scored_by_domain[domain].append(enriched)

        retrieved = []
        for domain in domain_order:
            ranked_all = sorted(
                scored_by_domain[domain],
                key=lambda item: item["reranker_score"],
                reverse=True,
            )
            quota = config["retrieval_domains"][domain]["quota"]
            if domain == "chemical_direction":
                ranked = []
                for required_type in config["retrieval_domains"][domain].get(
                    "required_knowledge_types", []
                ):
                    match = next(
                        (
                            item
                            for item in ranked_all
                            if item not in ranked
                            and chunks[item["idx"]].get("knowledge_type") == required_type
                        ),
                        None,
                    )
                    if match is not None:
                        ranked.append(match)
                for item in ranked_all:
                    if len(ranked) >= quota:
                        break
                    if item not in ranked:
                        ranked.append(item)
            elif domain == "site_or_enterprise":
                ranked = []
                predicates = [
                    lambda chunk: chunk.get("subject") == well["well_id"]
                    and chunk.get("knowledge_type") in {"研究井场地背景", "监测井背景"},
                    lambda chunk: chunk.get("subject") == well["well_id"]
                    and chunk.get("knowledge_type") == "井企候选关系",
                    lambda chunk: chunk.get("subject") in set(query_meta["candidate_enterprise_ids"])
                    and chunk.get("knowledge_type") == "企业背景",
                ]
                for predicate in predicates:
                    match = next(
                        (
                            item
                            for item in ranked_all
                            if item not in ranked and predicate(chunks[item["idx"]])
                        ),
                        None,
                    )
                    if match is not None:
                        ranked.append(match)
                for item in ranked_all:
                    if len(ranked) >= quota:
                        break
                    if item not in ranked:
                        ranked.append(item)
            else:
                ranked = ranked_all[:quota]
            for item in ranked:
                chunk = dict(chunks[item["idx"]])
                chunk.update(
                    {
                        "retrieval_domain": domain,
                        "domain_rank": len([x for x in retrieved if x["retrieval_domain"] == domain]) + 1,
                        "vector_rank": item["vector_rank"],
                        "bm25_rank": item["bm25_rank"],
                        "rrf_score": item["rrf_score"],
                        "reranker_score": item["reranker_score"],
                    }
                )
                retrieved.append(chunk)

        for final_rank, chunk in enumerate(retrieved, start=1):
            chunk["final_rank"] = final_rank

        outputs.append(
            {
                "well_id": well["well_id"],
                "queries": queries,
                "query_metadata": query_meta,
                "retrieval_configuration": {
                    "embedding_model": config["embedding_model"],
                    "embedding_revision": config.get("embedding_revision"),
                    "reranker_model": config["reranker_model"],
                    "reranker_revision": config.get("reranker_revision"),
                    "vector_top_k_per_domain": config["vector_top_k"],
                    "bm25_top_k_per_domain": config["bm25_top_k"],
                    "rrf_constant": config["rrf_constant"],
                    "rrf_shortlist_k_per_domain": config["rrf_shortlist_k_per_domain"],
                    "domain_quotas": {
                        key: value["quota"] for key, value in config["retrieval_domains"].items()
                    },
                    "final_top_k": config["final_top_k"],
                },
                "retrieved_chunks": retrieved,
            }
        )
        print(f"[{order:02d}/{len(wells)}] {well['well_id']}: {len(retrieved)} chunks", flush=True)

    write_jsonl(paths["retrieval_output"], outputs)
    log = {
        "schema_version": "1.1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.time() - started, 3),
        "n_wells": len(outputs),
        "retrieval_output_sha256": sha256(paths["retrieval_output"]),
        "corpus_sha256": sha256(paths["corpus"]),
        "config": config,
    }
    write_json(paths["retrieval_log"], log)


def choose_representative_wells(rows: list[dict]) -> list[str]:
    groups: dict[str, list[tuple[float, str]]] = {}
    for row in rows:
        p1 = float(row["query_metadata"]["eem_overall_probability"])
        p2 = float(row["query_metadata"]["uvvis_overall_probability"])
        margin = abs(p1 - 0.5) + abs(p2 - 0.5)
        groups.setdefault(row["query_metadata"]["model_status"], []).append((margin, row["well_id"]))
    selected = []
    for status in ("Both-high", "EEM-only high", "UV-Vis-only high", "Both-low"):
        if groups.get(status):
            selected.append(sorted(groups[status], reverse=True)[0][1])
    return selected


def evaluate(config: dict, paths: dict[str, Path]) -> None:
    import pandas as pd

    rows = read_jsonl(paths["retrieval_output"])
    expected_groups = {
        "spectral_interpretation": {"光谱特征解释", "解释规则", "UV-Vis实验条件"},
        "chemical_direction": {"化学指标解释", "风险标签映射"},
        "analytical_method": {"标准分析方法"},
        "site_or_enterprise": {"研究井场地背景", "监测井背景", "井企候选关系", "企业背景", "场地环境知识", "水文地质知识"},
    }

    audit = []
    type_counter = Counter()
    for row in rows:
        chunks = row["retrieved_chunks"]
        types = [chunk.get("knowledge_type") for chunk in chunks]
        type_counter.update(types)
        ids = [chunk.get("chunk_id") for chunk in chunks]
        source_complete = all(bool(chunk.get("source_ids")) for chunk in chunks)
        record = {
            "well_id": row["well_id"],
            "model_status": row["query_metadata"]["model_status"],
            "retrieved_n": len(chunks),
            "unique_chunk_n": len(set(ids)),
            "duplicate_chunk_n": len(ids) - len(set(ids)),
            "source_id_complete": source_complete,
            "knowledge_type_n": len(set(types)),
            "spectral_interpretation_present": any(x in expected_groups["spectral_interpretation"] for x in types),
            "chemical_direction_present": any(x in expected_groups["chemical_direction"] for x in types),
            "analytical_method_present": any(x in expected_groups["analytical_method"] for x in types),
            "site_or_enterprise_present": any(x in expected_groups["site_or_enterprise"] for x in types),
            "mean_reranker_score": float(np.mean([x["reranker_score"] for x in chunks])) if chunks else None,
            "min_reranker_score": float(np.min([x["reranker_score"] for x in chunks])) if chunks else None,
        }
        record["all_four_domains_present"] = all(
            record[key]
            for key in (
                "spectral_interpretation_present",
                "chemical_direction_present",
                "analytical_method_present",
                "site_or_enterprise_present",
            )
        )
        audit.append(record)

    paths["quality_csv"].parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(audit).to_csv(paths["quality_csv"], index=False, encoding="utf-8-sig")

    summary = {
        "schema_version": "1.0",
        "n_wells": len(rows),
        "all_wells_have_12_chunks": all(x["retrieved_n"] == config["final_top_k"] for x in audit),
        "all_chunk_ids_unique_within_well": all(x["duplicate_chunk_n"] == 0 for x in audit),
        "all_chunks_have_source_ids": all(x["source_id_complete"] for x in audit),
        "wells_with_all_four_domains": sum(x["all_four_domains_present"] for x in audit),
        "domain_coverage": {
            key: sum(
                any(chunk.get("knowledge_type") in values for chunk in row["retrieved_chunks"])
                for row in rows
            )
            for key, values in expected_groups.items()
        },
        "retrieved_knowledge_type_counts": dict(type_counter),
        "representative_wells": choose_representative_wells(rows),
        "note": "Domain coverage is a screening diagnostic, not a relevance gold standard. Representative-well rows require blinded human review before LLM inference.",
    }
    write_json(paths["quality_summary"], summary)

    selected = set(summary["representative_wells"])
    review_rows = []
    for row in rows:
        if row["well_id"] not in selected:
            continue
        for chunk in row["retrieved_chunks"]:
            review_rows.append(
                {
                    "well_id": row["well_id"],
                    "model_status": row["query_metadata"]["model_status"],
                    "final_rank": chunk["final_rank"],
                    "chunk_id": chunk["chunk_id"],
                    "entry_id": chunk["entry_id"],
                    "knowledge_type": chunk["knowledge_type"],
                    "subject": chunk["subject"],
                    "category": chunk["category"],
                    "content": chunk["content"],
                    "scope": chunk["scope"],
                    "limitations": chunk["limitations"],
                    "source_ids": ";".join(chunk["source_ids"]),
                    "reranker_score": chunk["reranker_score"],
                    "manual_relevance_0_2": "",
                    "manual_applicability_0_2": "",
                    "review_note": "",
                }
            )
    with pd.ExcelWriter(paths["manual_review_xlsx"], engine="openpyxl") as writer:
        pd.DataFrame(review_rows).to_excel(writer, index=False, sheet_name="Representative wells")
        pd.DataFrame(
            [
                ["manual_relevance_0_2", "0=irrelevant, 1=partly relevant, 2=directly relevant"],
                ["manual_applicability_0_2", "0=inapplicable, 1=conditional, 2=applicable"],
                ["review_note", "Record mismatch, missing context, conflict or reason for exclusion"],
            ],
            columns=["Field", "Definition"],
        ).to_excel(writer, index=False, sheet_name="Scoring guide")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "retrieve", "evaluate", "all"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    config = read_json(args.config)
    paths = resolve_paths(config)

    if args.command in ("build", "all"):
        build(config, paths)
    if args.command in ("retrieve", "all"):
        retrieve(config, paths)
    if args.command in ("evaluate", "all"):
        evaluate(config, paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
