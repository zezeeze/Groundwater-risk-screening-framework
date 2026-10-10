# Groundwater-risk-screening-framework
An explainable groundwater risk-screening framework integrating EEM and UV-Vis spectra, hierarchical machine learning, SHAP, hybrid RAG, and LLM-based evidence synthesis.
# GroundwaterSpectra-LLM

GroundwaterSpectra-LLM is a groundwater risk-screening framework integrating excitation–emission matrix fluorescence spectroscopy (EEM), UV–Vis absorption spectroscopy, hierarchical machine learning, SHAP interpretation, retrieval-augmented generation (RAG), and a local large language model.

The repository provides the anonymized data and code required to retrain the machine-learning models, reproduce the main analyses, construct the RAG knowledge base, and run the groundwater risk-screening application.

## Workflow

1. Extract structured features from EEM and UV–Vis spectra.
2. Train independent EEM and UV–Vis models for:
   - overall groundwater risk identification;
   - five-category chemical anomaly identification.
3. Calculate out-of-fold probabilities and SHAP contributions.
4. Retrieve relevant information from the anonymized RAG knowledge base.
5. Use the local LLM to generate:
   - risk-screening levels;
   - candidate confirmatory tests;
   - candidate enterprises for investigation;
   - candidate contaminant–enterprise links;
   - uncertainty statements.

## Repository structure

```text
Groundwater-risk-screening-framework/
├── data/
│   ├── chemical/               # Anonymized chemical measurements
│   ├── EEM/                    # EEM spectral data
│   ├── UV-Vis/                 # UV–Vis spectral data
│   ├── RAG_knowledge_base/     # Anonymized RAG documents and corpus
│   └── LLM_inputs_outputs/     # Structured inputs and evaluation files
├── code/
│   ├── machine_learning/       # Model training, SHAP, and evaluation
│   ├── RAG_LLM/                # Retrieval and local LLM pipeline
│   └── plotting/               # Figure-generation scripts
├── models/                     # Trained machine-learning models
├── app/                        # GroundwaterSpectra-LLM desktop application
├── figures/                    # Reproduced manuscript figures
└── requirements/               # Python dependency files
```

## Installation

Python 3.12 is recommended.

```bash
git clone https://github.com/zezeeze/Groundwater-risk-screening-framework.git
cd Groundwater-risk-screening-framework

python -m venv .venv
```

Activate the environment:

```bash
# Windows
.venv\Scripts\activate

# Linux or macOS
source .venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements/requirements_machine_learning.txt
pip install -r requirements/requirements_rag_llm.txt
```

## Model training

Run the scripts in the following order:

```bash
python code/machine_learning/train_nested_cv.py
python code/machine_learning/train_category_models.py
python code/machine_learning/compute_oof_shap.py
```

These scripts generate:

- out-of-fold overall-risk probabilities;
- five-category risk probabilities;
- trained EEM and UV–Vis models;
- well-level SHAP contributions;
- model-performance tables.

The EEM and UV–Vis paths are trained independently.

## RAG retrieval

After the machine-learning outputs have been generated, build the retrieval index and obtain well-specific knowledge snippets:

```bash
python code/RAG_LLM/rag_pipeline.py build
python code/RAG_LLM/rag_pipeline.py retrieve
python code/RAG_LLM/prepare_llm_requests.py
```

The RAG knowledge base contains anonymized information on:

- spectral-feature interpretation;
- pollutant properties and environmental behavior;
- groundwater standards and analytical methods;
- regional industrial activities;
- well–enterprise spatial relationships;
- enterprise products, raw materials, and characteristic pollutants.

## Local LLM

The LLM module uses:

- **Qwen3-8B-GGUF Q4_K_M**
- **llama.cpp**

Download the model separately and place it in:

```text
models/LLM/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf
```

Place the llama.cpp runtime in:

```text
models/LLM/llama.cpp/
```

Start the local inference server:

```powershell
powershell -ExecutionPolicy Bypass -File code/RAG_LLM/start_local_qwen_server.ps1
```

Then run the well-level inference:

```bash
python code/RAG_LLM/run_local_qwen.py --mode all
```

## Desktop application

The graphical application can be started with:

```bash
python app/groundwater_app_v2.py
```

The application supports spectral-data import, feature extraction, machine-learning prediction, SHAP interpretation, RAG retrieval, and LLM-assisted candidate generation.

## Data anonymization

All publicly available monitoring-well and enterprise identifiers are anonymized. Spatial coordinates are transformed using a consistent translation that preserves relative distances, directions, and spatial relationships. The correspondence with the original identifiers is not included in this repository.

Chemical measurements used as reference labels are separated from the LLM inference inputs and are used only for model training and independent evaluation.

## Citation

Citation information will be added after publication.
