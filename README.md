# GroundwaterSpectra-LLM

GroundwaterSpectra-LLM is a reproducible groundwater risk-screening workflow that combines excitation-emission matrix fluorescence spectroscopy (EEM), UV-Vis absorption spectroscopy, hierarchical machine learning, SHAP interpretation, hybrid retrieval-augmented generation (RAG), and a local large language model (LLM).

The repository provides anonymized data, source code, trained machine-learning models, fixed RAG/LLM inputs, frozen LLM outputs, and evaluation files for the 41-well study. The desktop application and rendered manuscript figures are not included. A new LLM run is printed in the terminal and saved as JSONL, while the published frozen output remains available for reproducing the reported evaluation.

## Workflow

1. Chemical measurements define the overall-risk label and five chemical-anomaly labels.
2. EEM and UV-Vis features are modeled as two independent evidence paths.
3. The first model layer predicts overall groundwater risk.
4. The second layer predicts five chemical-anomaly categories.
5. Out-of-fold SHAP values identify the spectral features supporting each prediction.
6. Hybrid RAG retrieves relevant spectral, chemical, analytical-method, hydrogeological, and enterprise records.
7. A local Qwen3-8B model organizes the evidence into risk levels, confirmatory-test candidates, enterprise candidates, contaminant-enterprise links, and uncertainty statements.

## Repository structure

All paths below are relative to the repository root.

```text
Groundwater-risk-screening-framework/
├── analysis/
│   └── machine_learning/             # Training and evaluation code and outputs
├── code/
│   ├── rag_llm/                      # RAG, local LLM and evaluation pipeline
│   └── merge_eem_files.py            # EEM file-merging utility
├── data/
│   ├── chemical/                     # Anonymized chemical measurements
│   ├── eem/                          # Raw and blank-corrected EEM files
│   ├── uv_vis/                       # Blank-corrected UV-Vis spectra
│   ├── model_inputs/                 # Released machine-learning input tables
│   ├── rag/                          # Anonymized knowledge base and spatial files
│   └── llm/                          # Structured inputs, prompts, outputs and evaluation
├── models/
│   ├── machine_learning/             # Released fitted models
│   └── llm/                          # Location for downloaded LLM runtime files
├── docs/                             # Environment and runtime requirements
├── README.md
└── SHA256SUMS.csv
```

The knowledge-base text, prompt, and frozen LLM outputs retain Chinese content because the reported inference experiment was conducted in Chinese. Repository paths and executable commands use English names.

## Requirements

- Python 3.12
- Git
- PowerShell on Windows for the supplied llama.cpp launch script
- At least 16 GB system memory
- An NVIDIA GPU and a compatible CUDA build of llama.cpp are recommended for local Qwen3-8B inference

The machine-learning and RAG stages can run without the Qwen model. The LLM stage requires the GGUF model and llama.cpp runtime described below.

## 1. Download the repository

```powershell
git clone https://github.com/zezeeze/Groundwater-risk-screening-framework.git
cd Groundwater-risk-screening-framework
```

Run every command in this README from the repository root.

## 2. Create the machine-learning environment

Two environments are used because the released machine-learning and RAG environments contain different package versions.

```powershell
py -3.12 -m venv .venv-ml
.\.venv-ml\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r "docs/requirements_machine_learning.txt"
deactivate
```

## 3. Reproduce the machine-learning analysis

Activate the machine-learning environment:

```powershell
.\.venv-ml\Scripts\Activate.ps1
```

The two principal model-input workbooks are:

```text
data/model_inputs/chemical_reference_41_wells.xlsx
data/model_inputs/spectral_features_41_wells.xlsx
```

### 3.1 Overall-risk models

```powershell
python "analysis/machine_learning/code/train_nested_cv.py"
```

The script performs nested leave-one-well-out cross-validation and writes overall-risk metrics, out-of-fold probabilities, labels, selected parameters, and run summaries to:

```text
analysis/machine_learning/results/
```

Fitted model bundles are written to `models/machine_learning/`.

### 3.2 Five-category models

```powershell
python "analysis/machine_learning/code/train_category_models.py"
```

The script writes category probabilities, performance metrics, SHAP values, and category definitions to:

```text
analysis/machine_learning/data/
```

The retained category-model bundle is written to `models/machine_learning/`.

### 3.3 Overall-risk SHAP values

```powershell
python "analysis/machine_learning/code/compute_oof_shap.py"
```

This step calculates out-of-fold SHAP contributions for the retained EEM and UV-Vis overall-risk models.

## 4. Create the RAG/LLM environment

```powershell
py -3.12 -m venv .venv-rag
.\.venv-rag\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r "code/rag_llm/requirements_frozen.txt"
```

Download the fixed retrieval-model revisions into the local Hugging Face cache:

```powershell
python -c "from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-m3', revision='5617a9f61b028005a4858fdac845db406aefb181'); snapshot_download('BAAI/bge-reranker-v2-m3', revision='953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e')"
```

## 5. Build and check the RAG retrieval layer

The principal public inputs are:

```text
data/rag/rag_corpus.jsonl
data/llm/structured_inputs_41_wells.jsonl
data/llm/category_model_performance_metadata.json
data/llm/prompt_template_and_output_format.md
data/llm/output_schema.json
code/rag_llm/rag_config.json
code/rag_llm/local_qwen_config.json
```

If the category models were retrained, refresh their shared validation metadata. The script automatically uses the newly generated summary in `analysis/machine_learning/data/`; otherwise, it uses the released study summary in `analysis/machine_learning/results/`.

```powershell
python "code/rag_llm/build_shared_model_metadata.py"
```

Build the FAISS and BM25 indexes, retrieve the well-specific evidence, and run the retrieval checks:

```powershell
python "code/rag_llm/rag_pipeline.py" all
```

The main generated files are:

```text
data/rag/index/
data/llm/rag_retrieval_41_wells.jsonl
data/llm/rag_run_log.json
data/llm/rag_quality_summary.json
```

Assemble the provider-neutral LLM requests:

```powershell
python "code/rag_llm/prepare_llm_requests.py"
```

The request file is written to:

```text
data/llm/inference_requests_41_wells.jsonl
```

The supplied `structured_inputs_41_wells.jsonl` is the fixed interface used in the reported 41-well experiment. It contains the released spectral features, out-of-fold probabilities, SHAP evidence, and anonymized site context. Chemical reference answers are stored separately and are not read during retrieval or inference.

## 6. Install the local LLM

The reported implementation used:

- repository: `Qwen/Qwen3-8B-GGUF`
- revision: `7c41481f57cb95916b40956ab2f0b139b296d974`
- model file: `Qwen3-8B-Q4_K_M.gguf`
- llama.cpp: `b11026`

Download the GGUF file:

```powershell
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='Qwen/Qwen3-8B-GGUF', filename='Qwen3-8B-Q4_K_M.gguf', revision='7c41481f57cb95916b40956ab2f0b139b296d974', local_dir='models/llm/Qwen3-8B-GGUF')"
```

Download a compatible llama.cpp release from <https://github.com/ggml-org/llama.cpp/releases>.

For the supplied Windows CUDA launch script, use the following locations:

```text
models/llm/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf
models/llm/llama.cpp-b11026-cuda12.4/bin/llama-server.exe
```

For a different llama.cpp build, GPU, or CPU backend, edit `code/rag_llm/start_local_qwen_server.ps1`.

## 7. Run the LLM in the terminal

Open PowerShell terminal 1 and start the server:

```powershell
powershell -ExecutionPolicy Bypass -File "code/rag_llm/start_local_qwen_server.ps1"
```

Keep terminal 1 open. After the server begins listening on `127.0.0.1:8088`, open terminal 2 and activate the RAG environment:

```powershell
.\.venv-rag\Scripts\Activate.ps1
```

Test the four representative wells first. The explicit output directory keeps a new run separate from the published frozen output:

```powershell
python "code/rag_llm/run_local_qwen.py" --config "code/rag_llm/local_qwen_config.json" --mode representative --output-dir "data/llm/local_llm_run_reproduced"
```

If the test succeeds, process all 41 wells:

```powershell
python "code/rag_llm/run_local_qwen.py" --config "code/rag_llm/local_qwen_config.json" --mode all --output-dir "data/llm/local_llm_run_reproduced"
```

Each validated well-level JSON record is printed directly in terminal 2. The same records are saved under:

```text
data/llm/local_llm_run_reproduced/
```

Resume an interrupted run with:

```powershell
python "code/rag_llm/run_local_qwen.py" --config "code/rag_llm/local_qwen_config.json" --mode all --resume --output-dir "data/llm/local_llm_run_reproduced"
```

## 8. Reproduce the published LLM evaluation

The files under `data/llm/local_llm_run/` are the published, frozen study output and its evaluation records. The independent chemical and investigation references remain outside the inference input. The evaluation script verifies the frozen-output hash before opening the reference answers; it does not require the local Qwen server.

```powershell
python "code/rag_llm/evaluate_local_llm.py" `
  --outputs "data/llm/local_llm_run/all_wells_qwen3_8b_outputs.jsonl" `
  --references "data/llm/independent_evaluation_answers_41_wells.json" `
  --inputs "data/llm/structured_inputs_41_wells.jsonl" `
  --freeze-manifest "data/llm/local_llm_run/all_wells_qwen3_8b_frozen_manifest.json" `
  --output-dir "data/llm/published_evaluation_reproduced"
```

The evaluation covers risk-category ranking, confirmatory-test candidates, enterprise candidates, contaminant-enterprise links, required-field completeness, and evidence-reference validity.

To evaluate a newly generated 41-well output, freeze it before opening the independent reference file, then pass the new manifest and output paths to the same evaluator:

```powershell
python "code/rag_llm/freeze_llm_output.py" `
  --outputs "data/llm/local_llm_run_reproduced/all_wells_qwen3_8b_outputs.jsonl" `
  --manifest "data/llm/local_llm_run_reproduced/all_wells_qwen3_8b_frozen_manifest.json"

python "code/rag_llm/evaluate_local_llm.py" `
  --outputs "data/llm/local_llm_run_reproduced/all_wells_qwen3_8b_outputs.jsonl" `
  --references "data/llm/independent_evaluation_answers_41_wells.json" `
  --inputs "data/llm/structured_inputs_41_wells.jsonl" `
  --freeze-manifest "data/llm/local_llm_run_reproduced/all_wells_qwen3_8b_frozen_manifest.json" `
  --output-dir "data/llm/local_llm_run_reproduced/evaluation_results"
```

## 9. Use the workflow with a new dataset

1. Format the chemical and spectral variables using the released workbooks as templates.
2. Retain the same 11 EEM and 13 UV-Vis feature definitions, or document any changes.
3. Retrain and validate the overall-risk and category models using the scripts in `analysis/machine_learning/code/`.
4. Create one structured JSONL record per well following the field structure in `data/llm/structured_inputs_41_wells.jsonl`.
5. Replace the regional records in `data/rag/` with an anonymized knowledge base for the new site.
6. Update `code/rag_llm/rag_config.json` if file locations or retrieval settings change.
7. Rebuild retrieval, prepare the inference requests, and run the local LLM in a new output directory.
8. If independent reference answers are available, freeze the new output before running the evaluation script.

The released scripts reproduce the reported 41-well experiment. A new site requires local reference samples for model validation and a site-specific knowledge-base update.

## 10. Anonymization

- Monitoring wells use the anonymized identifiers `CK`, `A`, `B`, `C`, and `D`.
- Historical monitoring records that are not part of the 41-well modeling set use `H` identifiers.
- Enterprises use the anonymized identifiers `E001-E060`.
- The original identifier correspondence is not included.
- Spatial coordinates were shifted by a common constant, preserving relative distance, direction, point-in-polygon relationships, and regional-flow comparisons.
- The public RAG corpus contains only anonymized site and enterprise information.

## 11. File-integrity check

`SHA256SUMS.csv` records the relative path, SHA-256 value, and file size for every released file except the checksum table itself. It can be used to verify that files were not altered during transfer.

## 12. Common problems

### The RAG pipeline cannot find BGE-M3

Run the retrieval-model download command in Section 4 before running `rag_pipeline.py`.

### The LLM script cannot connect to port 8088

Confirm that `llama-server` is still running in terminal 1 and that `code/rag_llm/local_qwen_config.json` contains:

```json
"endpoint": "http://127.0.0.1:8088"
```

### CUDA is unavailable

Use a llama.cpp build compatible with the local hardware and edit the launch script. Machine-learning training and RAG indexing do not require llama.cpp.

### A file cannot be found

Run commands from the repository root and keep the released directory structure unchanged.

## Citation

Citation information will be added after publication.
