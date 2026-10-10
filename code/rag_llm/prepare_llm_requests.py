from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "llm"

WELL_INPUTS = DATA_DIR / "structured_inputs_41_wells.jsonl"
RAG_RESULTS = DATA_DIR / "rag_retrieval_41_wells.jsonl"
SHARED_METADATA = DATA_DIR / "category_model_performance_metadata.json"
PROMPT_FILE = DATA_DIR / "prompt_template_and_output_format.md"
OUTPUT_SCHEMA = DATA_DIR / "output_schema.json"
OUTPUT = DATA_DIR / "inference_requests_41_wells.jsonl"
LOG = DATA_DIR / "request_preparation_log.json"


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def extract_system_prompt(markdown: str) -> str:
    start = markdown.index("## 1. 系统提示词") + len("## 1. 系统提示词")
    end = markdown.index("## 2. 逐井用户提示词模板")
    prompt = markdown[start:end].strip()
    return prompt + "\n\n输出必须通过随请求提供的JSON Schema验证。"


def main() -> None:
    wells = load_jsonl(WELL_INPUTS)
    rag = {item["well_id"]: item for item in load_jsonl(RAG_RESULTS)}
    shared = json.loads(SHARED_METADATA.read_text(encoding="utf-8"))
    schema = json.loads(OUTPUT_SCHEMA.read_text(encoding="utf-8"))
    system_prompt = extract_system_prompt(PROMPT_FILE.read_text(encoding="utf-8"))

    if len(wells) != 41 or len(rag) != 41:
        raise ValueError(f"Expected 41 wells, found inputs={len(wells)}, retrieval={len(rag)}")

    task_instruction = (
        "分析当前监测井的结构化证据。保留EEM与UV-Vis两条路径及其分歧，"
        "结合类别模型性能确定证据权重。风险类别按两路概率和证据强度排序，"
        "确认性检测建议必须引用已定义的证据编号。候选污染物—企业组合必须同时具有"
        "企业生产资料、风险类别及井企关系证据。仅返回符合JSON Schema的结果。"
    )

    requests = []
    for well in wells:
        well_id = well["well_id"]
        if well_id not in rag:
            raise KeyError(f"Missing RAG result for {well_id}")
        rag_item = rag[well_id]
        if len(rag_item["retrieved_chunks"]) != 12:
            raise ValueError(f"{well_id} does not have 12 retrieved chunks")

        package = copy.deepcopy(well)
        package["rag_retrieval"] = {
            "status": "retrieved_and_frozen",
            "queries": rag_item["queries"],
            "retrieval_configuration": rag_item["retrieval_configuration"],
            "retrieved_chunks": rag_item["retrieved_chunks"],
        }
        requests.append(
            {
                "request_schema_version": "1.0",
                "request_id": f"GW-{well_id}",
                "well_id": well_id,
                "system_instruction": system_prompt,
                "task_instruction": task_instruction,
                "well_evidence_package": package,
                "shared_category_model_performance": shared,
                "output_schema": schema,
                "excluded_reference_files": [
                    "independent_evaluation_answers_41_wells.json",
                    "independent_evaluation_answers_41_wells.xlsx",
                ],
            }
        )

    with OUTPUT.open("w", encoding="utf-8", newline="\n") as handle:
        for item in requests:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    log = {
        "schema_version": "1.0",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "n_requests": len(requests),
        "all_requests_have_12_rag_chunks": all(
            len(item["well_evidence_package"]["rag_retrieval"]["retrieved_chunks"]) == 12
            for item in requests
        ),
        "all_request_ids_unique": len({item["request_id"] for item in requests}) == len(requests),
        "reference_answers_loaded": False,
        "input_hashes": {
            path.name: sha256(path)
            for path in [WELL_INPUTS, RAG_RESULTS, SHARED_METADATA, PROMPT_FILE, OUTPUT_SCHEMA]
        },
        "output_sha256": sha256(OUTPUT),
    }
    LOG.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
