from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
OUTPUT = SCRIPT_DIR / "reproducibility_manifest.json"

MODEL_REVISIONS = {
    "embedding_model": {
        "model_id": "BAAI/bge-m3",
        "revision": "5617a9f61b028005a4858fdac845db406aefb181",
    },
    "reranker_model": {
        "model_id": "BAAI/bge-reranker-v2-m3",
        "revision": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
        "model_weights_sha256": "D9E3E081FAFF1EEFB84019509B2F5558FD74C1A05A2C7DB22F74174FCEDB5286",
    },
}

FILES = [
    "code/rag_llm/rag_pipeline.py",
    "code/rag_llm/rag_config.json",
    "code/rag_llm/README.md",
    "code/rag_llm/build_shared_model_metadata.py",
    "code/rag_llm/prepare_llm_requests.py",
    "code/rag_llm/requirements_frozen.txt",
    "data/rag/rag_corpus.jsonl",
    "data/rag/index/index_manifest.json",
    "data/rag/index/chunks.jsonl",
    "data/rag/index/embeddings.npy",
    "data/rag/index/faiss.index",
    "data/rag/index/bm25.pkl",
    "data/llm/structured_inputs_41_wells.jsonl",
    "data/llm/category_model_performance_metadata.json",
    "data/llm/prompt_template_and_output_format.md",
    "data/llm/output_schema.json",
    "data/llm/validate_output_evidence_references.py",
    "data/llm/rag_retrieval_41_wells.jsonl",
    "data/llm/rag_run_log.json",
    "data/llm/rag_quality_checks.csv",
    "data/llm/rag_quality_summary.json",
    "data/llm/rag_representative_well_review.xlsx",
    "data/llm/inference_requests_41_wells.jsonl",
    "data/llm/request_preparation_log.json",
]

PACKAGES = [
    "sentence-transformers",
    "transformers",
    "torch",
    "faiss-cpu",
    "rank-bm25",
    "jieba",
    "numpy",
    "pandas",
    "scikit-learn",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def main() -> None:
    file_records = []
    for relative in FILES:
        path = PROJECT_ROOT / relative
        if not path.exists():
            raise FileNotFoundError(path)
        file_records.append(
            {
                "path": relative.replace("\\", "/"),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )

    package_versions = {}
    for name in PACKAGES:
        try:
            package_versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            package_versions[name] = None

    manifest = {
        "schema_version": "1.0",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "python": {
            "version": sys.version.split()[0],
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
        },
        "package_versions": package_versions,
        "model_revisions": MODEL_REVISIONS,
        "random_seed": 20260921,
        "file_integrity": file_records,
        "notes": [
            "Chemical reference answers are not loaded by the retrieval pipeline.",
            "Manual relevance and applicability scores remain blank until blinded review.",
        ],
    }
    OUTPUT.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
