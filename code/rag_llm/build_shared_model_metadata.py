from __future__ import annotations

import json
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
RETRAINED_SOURCE = (
    PROJECT_ROOT
    / "analysis"
    / "machine_learning"
    / "data"
    / "category_training_summary.json"
)
RELEASED_SOURCE = (
    PROJECT_ROOT
    / "analysis"
    / "machine_learning"
    / "results"
    / "category_training_summary.json"
)
OUTPUT = PROJECT_ROOT / "data" / "llm" / "category_model_performance_metadata.json"


def main() -> None:
    # A new category-model run writes its summary to machine_learning/data.
    # The released, frozen study summary remains under results so the public
    # package can rebuild metadata without rerunning model training.
    source = RETRAINED_SOURCE if RETRAINED_SOURCE.exists() else RELEASED_SOURCE
    summary = json.loads(source.read_text(encoding="utf-8"))
    records = []
    for item in summary["metrics"]:
        records.append(
            {
                "spectral_path": item["Modality"],
                "category_id": item["Category"],
                "n_wells": summary["n_wells"],
                "positive_wells": item["Positive wells"],
                "prevalence": item["Prevalence"],
                "oof_pr_auc": item["PR-AUC"],
                "oof_pr_auc_95_ci": [item["PR-AUC low"], item["PR-AUC high"]],
                "oof_roc_auc": item["ROC-AUC"],
                "oof_roc_auc_95_ci": [item["ROC-AUC low"], item["ROC-AUC high"]],
            }
        )

    definitions = []
    for item in summary["categories"]:
        definitions.append(
            {
                "category_id": item["Category"],
                "category_name_zh": item["Chinese category"],
                # The delimiter is comma followed by a space. This preserves
                # compound names such as 1,2-Dichloroethane.
                "included_indicators": item["Included indicators"].split(", "),
                "positive_wells": item["Positive wells"],
                "label_rule": item["Rule"],
            }
        )

    output = {
        "schema_version": "1.0",
        "scope": "Shared category-model validation metadata supplied unchanged to every well-level LLM inference request.",
        "source_file": str(source.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "validation": "Nested leave-one-out cross-validation with outer-fold predictions; 95% confidence intervals from 5000 stratified bootstrap resamples.",
        "records": records,
        "category_definitions": definitions,
    }
    OUTPUT.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
