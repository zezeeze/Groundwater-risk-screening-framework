from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    hamming_loss,
    label_ranking_average_precision_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, LeaveOneOut, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SEED = 20260914
N_BOOT = 5000
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "analysis" / "machine_learning" / "data"
OUT.mkdir(parents=True, exist_ok=True)
MODEL_DIR = ROOT / "models" / "machine_learning"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

PROJECT = ROOT
CHEM_PATH = PROJECT / "data/model_inputs/chemical_reference_41_wells.xlsx"
SPEC_PATH = PROJECT / "data/model_inputs/spectral_features_41_wells.xlsx"

V_BOUNDARIES = {
    "Total hardness": 650.0, "TDS": 2000.0, "Sulfate": 350.0, "Chloride": 350.0,
    "Fe": 2.0, "Mn": 1.5, "Cu": 1.5, "Zn": 5.0, "Al": 0.5,
    "Volatile phenols": 0.01, "Anionic surfactants": 0.3, "CODMn": 10.0,
    "Ammonia-N": 1.5, "Sulfide": 0.1, "Na": 400.0, "Nitrite-N": 4.8,
    "Nitrate-N": 30.0, "Cyanide": 0.1, "Fluoride": 2.0, "Iodide": 0.5,
    "Hg": 0.002, "As": 0.05, "Se": 0.1, "Cd": 0.01, "Cr(VI)": 0.1,
    "Pb": 0.1, "B": 2.0, "Ni": 0.1, "Co": 0.1,
    "Dichloromethane": 500.0, "1,2-Dichloroethane": 40.0, "Vinyl chloride": 90.0,
    "Chlorobenzene": 600.0, "o-Dichlorobenzene": 2000.0, "Naphthalene": 600.0,
    "Benzo[a]pyrene": 0.5, "Benzene": 120.0, "Toluene": 1400.0,
    "Chloroform": 300.0, "Carbon tetrachloride": 50.0, "Xylene": 1000.0,
}

CATEGORY_DEFINITIONS = {
    "Conventional and nutrients": {
        "Chinese": "常规水质与营养盐",
        "Indicators": ["pH", "CODMn", "Ammonia-N", "Nitrite-N", "Nitrate-N"],
    },
    "Mineralization and major ions": {
        "Chinese": "盐分与矿化度",
        "Indicators": ["Total hardness", "TDS", "Sulfate", "Chloride", "Na", "Fluoride", "Iodide", "Sulfide"],
    },
    "Common metals": {
        "Chinese": "常见金属",
        "Indicators": ["Fe", "Mn", "Al"],
    },
    "Toxic metals and metalloids": {
        "Chinese": "毒性金属与类金属",
        "Indicators": ["Cu", "Zn", "Hg", "As", "Se", "Cd", "Cr(VI)", "Pb", "B", "Ni", "Co"],
    },
    "Specific organic pollutants": {
        "Chinese": "特征有机污染物",
        "Indicators": [
            "Volatile phenols", "Anionic surfactants", "Dichloromethane", "1,2-Dichloroethane",
            "Vinyl chloride", "Chlorobenzene", "o-Dichlorobenzene", "Naphthalene",
            "Benzo[a]pyrene", "Benzene", "Toluene", "Chloroform", "Carbon tetrachloride", "Xylene",
        ],
    },
}

MODALITIES = ["eem", "uv_vis"]


def load_data():
    chem = pd.read_excel(CHEM_PATH, sheet_name="Raw_2026")
    spec = pd.read_excel(SPEC_PATH, sheet_name="Spectral_features_24")
    definitions = pd.read_excel(SPEC_PATH, sheet_name="Definitions")
    ph = pd.read_excel(SPEC_PATH, sheet_name="Individual_chem_16", usecols=["Well", "pH"])
    chem = chem.merge(ph, on="Well", how="left", validate="one_to_one")

    flags = pd.DataFrame(index=chem.index)
    for indicator, boundary in V_BOUNDARIES.items():
        if indicator in chem.columns:
            flags[indicator] = pd.to_numeric(chem[indicator], errors="coerce") > boundary
    flags["pH"] = (pd.to_numeric(chem["pH"], errors="coerce") < 5.5) | (
        pd.to_numeric(chem["pH"], errors="coerce") > 9.0
    )
    chem["Overall risk"] = flags.any(axis=1).astype(int)

    category_labels = pd.DataFrame({"Well": chem["Well"]})
    for category, definition in CATEGORY_DEFINITIONS.items():
        columns = [x for x in definition["Indicators"] if x in flags.columns]
        category_labels[category] = flags[columns].any(axis=1).astype(int)
    category_labels["Overall risk"] = chem["Overall risk"].to_numpy()

    merged = spec.merge(category_labels, on="Well", how="inner", validate="one_to_one")
    if len(merged) != 41:
        raise RuntimeError(f"Expected 41 matched wells, got {len(merged)}")

    spectrum = definitions.set_index("Variable")["Spectrum"].astype(str)
    eem = [c for c in spec.columns if c != "Well" and spectrum.get(c, "").startswith("eem")]
    uv = [c for c in spec.columns if c != "Well" and spectrum.get(c, "").startswith("UV")]
    if len(eem) != 11 or len(uv) != 13:
        raise RuntimeError(f"Unexpected feature counts: EEM={len(eem)}, UV-Vis={len(uv)}")
    return merged, flags, {"eem": eem, "uv_vis": uv}, category_labels


def pipeline(seed):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            class_weight="balanced", solver="lbfgs", max_iter=5000, random_state=seed
        )),
    ])


PARAM_GRID = {"model__C": [0.01, 0.1, 1.0, 10.0]}


def binary_metrics(y, p):
    pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "Prevalence": float(np.mean(y)),
        "PR-AUC": average_precision_score(y, p),
        "ROC-AUC": roc_auc_score(y, p),
        "Brier": brier_score_loss(y, p),
        "Accuracy": accuracy_score(y, pred),
        "Balanced accuracy": balanced_accuracy_score(y, pred),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "Specificity": tn / (tn + fp),
        "F1": f1_score(y, pred, zero_division=0),
        "MCC": matthews_corrcoef(y, pred),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def stratified_ci(y, p, rng, n_boot=N_BOOT):
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    pr, roc = [], []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)])
        pr.append(average_precision_score(y[idx], p[idx]))
        roc.append(roc_auc_score(y[idx], p[idx]))
    return {
        "PR-AUC low": np.percentile(pr, 2.5), "PR-AUC high": np.percentile(pr, 97.5),
        "ROC-AUC low": np.percentile(roc, 2.5), "ROC-AUC high": np.percentile(roc, 97.5),
    }


def paired_delta_ci(y, a, b, rng, metric, n_boot=N_BOOT):
    scorer = average_precision_score if metric == "PR-AUC" else roc_auc_score
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    values = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)])
        values.append(scorer(y[idx], a[idx]) - scorer(y[idx], b[idx]))
    return np.percentile(values, [2.5, 97.5])


def fold_log_odds_shap(best_pipeline, x_train, x_test, feature_names):
    imputer = best_pipeline.named_steps["imputer"]
    scaler = best_pipeline.named_steps["scaler"]
    model = best_pipeline.named_steps["model"]
    z_train = scaler.transform(imputer.transform(x_train))
    z_test = scaler.transform(imputer.transform(x_test))[0]
    z_ref = z_train.mean(axis=0)
    coef = model.coef_[0]
    values = coef * (z_test - z_ref)
    base_logit = float(model.intercept_[0] + np.dot(coef, z_ref))
    prediction = float(expit(base_logit + values.sum()))
    return values, z_test, base_logit, prediction


def multilabel_summary(y_true, probability, modality):
    prediction = (probability >= 0.5).astype(int)
    risk_mask = y_true.sum(axis=1) > 0
    out = {
        "Modality": modality,
        "Micro PR-AUC": average_precision_score(y_true, probability, average="micro"),
        "Macro PR-AUC": average_precision_score(y_true, probability, average="macro"),
        "Micro ROC-AUC": roc_auc_score(y_true, probability, average="micro"),
        "Macro ROC-AUC": roc_auc_score(y_true, probability, average="macro"),
        "LRAP": label_ranking_average_precision_score(y_true[risk_mask], probability[risk_mask]),
        "Hamming loss": hamming_loss(y_true, prediction),
        "Sample F1": f1_score(y_true, prediction, average="samples", zero_division=0),
        "Exact match": np.mean(np.all(y_true == prediction, axis=1)),
    }
    for k in [1, 2, 3]:
        order = np.argsort(-probability[risk_mask], axis=1)[:, :k]
        hits = np.take_along_axis(y_true[risk_mask], order, axis=1)
        out[f"Top-{k} hit rate"] = np.mean(hits.max(axis=1))
        out[f"Top-{k} category recall"] = np.mean(hits.sum(axis=1) / y_true[risk_mask].sum(axis=1))
    return out


def frequency_baseline(y_true):
    risk_mask = y_true.sum(axis=1) > 0
    prevalence = y_true[risk_mask].mean(axis=0)
    probability = np.tile(prevalence, (risk_mask.sum(), 1))
    out = {
        "Modality": "Frequency baseline",
        "Micro PR-AUC": average_precision_score(y_true[risk_mask], probability, average="micro"),
        "Macro PR-AUC": average_precision_score(y_true[risk_mask], probability, average="macro"),
        "Micro ROC-AUC": roc_auc_score(y_true[risk_mask], probability, average="micro"),
        "Macro ROC-AUC": roc_auc_score(y_true[risk_mask], probability, average="macro"),
        "LRAP": label_ranking_average_precision_score(y_true[risk_mask], probability),
        "Hamming loss": np.nan, "Sample F1": np.nan, "Exact match": np.nan,
    }
    for k in [1, 2, 3]:
        order = np.argsort(-probability, axis=1)[:, :k]
        hits = np.take_along_axis(y_true[risk_mask], order, axis=1)
        out[f"Top-{k} hit rate"] = np.mean(hits.max(axis=1))
        out[f"Top-{k} category recall"] = np.mean(hits.sum(axis=1) / y_true[risk_mask].sum(axis=1))
    return out


def main():
    warnings.filterwarnings("ignore")
    data, flags, feature_sets, labels = load_data()
    categories = list(CATEGORY_DEFINITIONS)
    loo = LeaveOneOut()
    rng = np.random.default_rng(SEED)
    oof = labels.copy()
    metrics_rows, parameter_rows, shap_rows, additivity_rows = [], [], [], []
    retained = {}

    print("Category positives:")
    for category in categories:
        print(f"  {category}: {int(labels[category].sum())}/41")

    for modality_index, modality in enumerate(MODALITIES):
        X = data[feature_sets[modality]].copy()
        retained[modality] = {}
        for category_index, category in enumerate(categories):
            y = labels[category].to_numpy(int)
            probability = np.full(len(data), np.nan)
            for fold, (train_idx, test_idx) in enumerate(loo.split(X, y)):
                seed = SEED + 10000 * modality_index + 1000 * category_index + fold
                min_class = int(np.bincount(y[train_idx]).min())
                inner = StratifiedKFold(n_splits=min(5, min_class), shuffle=True, random_state=seed)
                search = GridSearchCV(
                    clone(pipeline(seed)), PARAM_GRID, scoring="average_precision", cv=inner,
                    n_jobs=-1, refit=True, error_score="raise"
                )
                search.fit(X.iloc[train_idx], y[train_idx])
                p = float(search.predict_proba(X.iloc[test_idx])[:, 1][0])
                probability[test_idx[0]] = p
                parameter_rows.append({
                    "Modality": modality, "Category": category,
                    "Held-out well": data.iloc[test_idx[0]]["Well"],
                    "Inner best PR-AUC": float(search.best_score_),
                    "Best parameters": json.dumps(search.best_params_, sort_keys=True),
                })

                values, z_test, base_logit, p_from_shap = fold_log_odds_shap(
                    search.best_estimator_, X.iloc[train_idx], X.iloc[test_idx], feature_sets[modality]
                )
                additivity_rows.append({
                    "Well": data.iloc[test_idx[0]]["Well"], "Modality": modality, "Category": category,
                    "OOF probability": p, "SHAP reconstructed probability": p_from_shap,
                    "Absolute difference": abs(p - p_from_shap), "Base log-odds": base_logit,
                    "Predicted log-odds": float(logit(np.clip(p, 1e-12, 1 - 1e-12))),
                    "SHAP sum": float(values.sum()),
                })
                for feature, value, standardized in zip(feature_sets[modality], values, z_test):
                    shap_rows.append({
                        "Well": data.iloc[test_idx[0]]["Well"], "Modality": modality,
                        "Category": category, "Feature": feature, "SHAP log-odds": float(value),
                        "Absolute SHAP": float(abs(value)), "Standardized feature value": float(standardized),
                    })

            oof[f"{modality}__{category}"] = probability
            row = {"Modality": modality, "Category": category, "Positive wells": int(y.sum()), **binary_metrics(y, probability)}
            row.update(stratified_ci(y, probability, rng))
            metrics_rows.append(row)
            print(f"{modality:6s} | {category:31s} PR={row['PR-AUC']:.3f} ROC={row['ROC-AUC']:.3f}")

            final_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + 90000 + 100 * modality_index + category_index)
            final_search = GridSearchCV(
                pipeline(SEED + 90000 + 100 * modality_index + category_index), PARAM_GRID,
                scoring="average_precision", cv=final_inner, n_jobs=-1, refit=True
            )
            final_search.fit(X, y)
            retained[modality][category] = {
                "model": final_search.best_estimator_, "features": feature_sets[modality],
                "best_parameters": final_search.best_params_, "positive_wells": int(y.sum()),
                "threshold": 0.5,
            }

    metrics = pd.DataFrame(metrics_rows)
    parameters = pd.DataFrame(parameter_rows)
    shap = pd.DataFrame(shap_rows)
    additivity = pd.DataFrame(additivity_rows)

    delta_rows = []
    for category in categories:
        y = labels[category].to_numpy(int)
        a = oof[f"EEM__{category}"].to_numpy()
        b = oof[f"UV-Vis__{category}"].to_numpy()
        for metric in ["PR-AUC", "ROC-AUC"]:
            low, high = paired_delta_ci(y, a, b, rng, metric)
            scorer = average_precision_score if metric == "PR-AUC" else roc_auc_score
            delta_rows.append({
                "Category": category, "Metric": metric,
                "EEM minus UV-Vis": scorer(y, a) - scorer(y, b),
                "95% CI low": low, "95% CI high": high,
            })
    deltas = pd.DataFrame(delta_rows)

    # Secondary evaluation within the 27 chemically defined risk wells. This
    # tests whether category probabilities distinguish risk composition rather
    # than only separating risk from non-risk wells.
    risk_mask = labels["Overall risk"].to_numpy(int) == 1
    risk_well_rows = []
    for modality in MODALITIES:
        for category in categories:
            y_risk = labels.loc[risk_mask, category].to_numpy(int)
            p_risk = oof.loc[risk_mask, f"{modality}__{category}"].to_numpy(float)
            risk_well_rows.append({
                "Modality": modality, "Category": category,
                "Risk wells": int(risk_mask.sum()), "Positive risk wells": int(y_risk.sum()),
                "Prevalence among risk wells": float(y_risk.mean()),
                "PR-AUC among risk wells": average_precision_score(y_risk, p_risk),
                "ROC-AUC among risk wells": roc_auc_score(y_risk, p_risk),
            })
    risk_well_metrics = pd.DataFrame(risk_well_rows)

    y_matrix = labels[categories].to_numpy(int)
    multilabel_rows = []
    for modality in MODALITIES:
        probability = np.column_stack([oof[f"{modality}__{x}"] for x in categories])
        multilabel_rows.append(multilabel_summary(y_matrix, probability, modality))
    multilabel_rows.append(frequency_baseline(y_matrix))
    multilabel = pd.DataFrame(multilabel_rows)

    global_shap = (
        shap.groupby(["Modality", "Category", "Feature"], as_index=False)
        .agg(**{"Mean absolute SHAP": ("Absolute SHAP", "mean"), "Mean signed SHAP": ("SHAP log-odds", "mean")})
    )
    global_shap["Relative contribution (%)"] = global_shap.groupby(["Modality", "Category"])["Mean absolute SHAP"].transform(lambda x: 100 * x / x.sum())
    global_shap["Rank"] = global_shap.groupby(["Modality", "Category"])["Mean absolute SHAP"].rank(method="first", ascending=False).astype(int)

    category_definitions = []
    for category, definition in CATEGORY_DEFINITIONS.items():
        category_definitions.append({
            "Category": category, "Chinese category": definition["Chinese"],
            "Included indicators": ", ".join(definition["Indicators"]),
            "Positive wells": int(labels[category].sum()),
            "Rule": "1 if at least one included indicator reaches Class V; otherwise 0",
        })

    labels.to_csv(OUT / "category_labels.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(category_definitions).to_csv(OUT / "category_definitions.csv", index=False, encoding="utf-8-sig")
    oof.to_csv(OUT / "category_oof_probabilities.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(OUT / "category_model_metrics.csv", index=False, encoding="utf-8-sig")
    multilabel.to_csv(OUT / "category_multilabel_metrics.csv", index=False, encoding="utf-8-sig")
    deltas.to_csv(OUT / "category_paired_differences.csv", index=False, encoding="utf-8-sig")
    risk_well_metrics.to_csv(OUT / "category_risk_well_discrimination.csv", index=False, encoding="utf-8-sig")
    parameters.to_csv(OUT / "category_inner_cv_parameters.csv", index=False, encoding="utf-8-sig")
    shap.to_csv(OUT / "category_oof_shap_long.csv", index=False, encoding="utf-8-sig")
    global_shap.to_csv(OUT / "category_shap_importance.csv", index=False, encoding="utf-8-sig")
    additivity.to_csv(OUT / "category_shap_additivity.csv", index=False, encoding="utf-8-sig")

    joblib.dump({
        "models": retained, "category_definitions": CATEGORY_DEFINITIONS,
        "risk_boundaries": {**V_BOUNDARIES, "pH": "<5.5 or >9.0"},
        "model_role": "parallel category probabilities for evidence prioritization",
    }, MODEL_DIR / "category_models.joblib")

    summary = {
        "seed": SEED, "n_wells": 41, "categories": category_definitions,
        "metrics": metrics.to_dict(orient="records"),
        "multilabel_metrics": multilabel.to_dict(orient="records"),
        "max_shap_additivity_error": float(additivity["Absolute difference"].max()),
    }
    (OUT / "category_training_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
