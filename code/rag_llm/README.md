# RAG and local LLM pipeline

This directory contains the hybrid retrieval, local inference, and evaluation code used by GroundwaterSpectra-LLM. The repository-level `README.md` is the authoritative step-by-step guide.

The current workflow is:

1. `build_shared_model_metadata.py` refreshes category-model validation metadata after model retraining.
2. `rag_pipeline.py all` builds the FAISS and BM25 indexes, performs reciprocal-rank fusion and BGE reranking, and retrieves 12 evidence chunks per well.
3. `prepare_llm_requests.py` combines the frozen well evidence, retrieval results, prompt, and JSON Schema without loading independent evaluation answers.
4. `start_local_qwen_server.ps1` starts Qwen3-8B-GGUF through llama.cpp.
5. `run_local_qwen.py` generates and validates the well-level JSON records. It returns five ranked risk categories, ten confirmatory-test candidates, five enterprise candidates, ten contaminant-enterprise links, and traceable evidence fields.
6. `freeze_llm_output.py` records the hash of a completed output before independent reference evaluation.
7. `evaluate_local_llm.py` evaluates the frozen output against the held-out chemical and investigation references.

Use `--output-dir` with `run_local_qwen.py` so a new run does not overwrite the published output under `data/llm/local_llm_run/`.

The reported retrieval configuration uses 400-token chunks with 80-token overlap, BGE-M3 embeddings, normalized-vector FAISS search, BM25 retrieval, reciprocal-rank fusion with constant 60, and BGE-reranker-v2-m3. Retrieval is divided into spectral interpretation, chemical direction, analytical methods, and site or enterprise context, with three retained chunks per domain.

Model and package revisions are recorded in `local_qwen_config.json`, `requirements_frozen.txt`, and `docs/llm_model_runtime.md`.
