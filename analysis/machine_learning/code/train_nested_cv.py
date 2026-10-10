from __future__ import annotations

import json
import os
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import loguniform
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, LeaveOneOut, RandomizedSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier


SEED = 20260909
N_BOOT = 5000
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "analysis" / "machine_learning" / "results"
OUT.mkdir(parents=True, exist_ok=True)
MODEL_DIR = ROOT / "models" / "machine_learning"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

CHEM_PATH = ROOT / "data" / "model_inputs" / "chemical_reference_41_wells.xlsx"
SPEC_PATH = ROOT / "data" / "model_inputs" / "spectral_features_41_wells.xlsx"

# IV-class upper boundaries. A value above a boundary is assigned to Class V.
# Color and turbidity are intentionally excluded from the chemical-risk endpoint.
V_BOUNDARIES = {
    "Total hardness": 650.0,
    "TDS": 2000.0,
    "Sulfate": 350.0,
    "Chloride": 350.0,
    "Fe": 2.0,
    "Mn": 1.5,
    "Cu": 1.5,
    "Zn": 5.0,
    "Al": 0.5,
    "Volatile phenols": 0.01,
    "Anionic surfactants": 0.3,
    "CODMn": 10.0,
    "Ammonia-N": 1.5,
    "Sulfide": 0.1,
    "Na": 400.0,
    "Nitrite-N": 4.8,
    "Nitrate-N": 30.0,
    "Cyanide": 0.1,
    "Fluoride": 2.0,
    "Iodide": 0.5,
    "Hg": 0.002,
    "As": 0.05,
    "Se": 0.1,
    "Cd": 0.01,
    "Cr(VI)": 0.1,
    "Pb": 0.1,
    "B": 2.0,
    "Ni": 0.1,
    "Co": 0.1,
    # Organic compounds are recorded in ug/L in the raw laboratory workbook.
    "Dichloromethane": 500.0,
    "1,2-Dichloroethane": 40.0,
    "Vinyl chloride": 90.0,
    "Chlorobenzene": 600.0,
    "o-Dichlorobenzene": 2000.0,
    "Naphthalene": 600.0,
    "Benzo[a]pyrene": 0.5,
    "Benzene": 120.0,
    "Toluene": 1400.0,
    "Chloroform": 300.0,
    "Carbon tetrachloride": 50.0,
    "Xylene": 1000.0,
}

UG_L_INDICATORS = {
    "Dichloromethane", "1,2-Dichloroethane", "Vinyl chloride", "Chlorobenzene",
    "o-Dichlorobenzene", "Naphthalene", "Benzo[a]pyrene", "Benzene", "Toluene",
    "Chloroform", "Carbon tetrachloride", "Xylene",
}

MODEL_ORDER = [
    "Logistic",
    "DT",
    "RF",
    "ET",
    "GBDT",
    "XGBoost",
    "SVM-RBF",
    "KNN",
    "sLDA",
    "GNB",
]
JOINT_INPUT = "EEM+UV-Vis"
MODALITY_ORDER = ["EEM-only", "UV-Vis-only", JOINT_INPUT]


def scaled_pipeline(estimator):
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", estimator),
        ]
    )


def tree_pipeline(estimator):
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", estimator)])


def model_specs(seed: int):
    return {
        "Logistic": {
            "pipeline": scaled_pipeline(
                LogisticRegression(class_weight="balanced", solver="liblinear", max_iter=5000, random_state=seed)
            ),
            "params": {"model__C": [0.01, 0.1, 1.0, 10.0, 100.0], "model__penalty": ["l1", "l2"]},
            "search": "grid",
        },
        "DT": {
            "pipeline": tree_pipeline(DecisionTreeClassifier(class_weight="balanced", random_state=seed)),
            "params": {
                "model__max_depth": [2, 3, 4, 5, None],
                "model__min_samples_leaf": [1, 2, 4],
                "model__criterion": ["gini", "entropy"],
            },
            "search": "grid",
        },
        "RF": {
            "pipeline": tree_pipeline(
                RandomForestClassifier(
                    n_estimators=300,
                    class_weight="balanced",
                    random_state=seed,
                    n_jobs=1,
                )
            ),
            "params": {
                "model__max_depth": [None, 3, 5, 8],
                "model__max_features": ["sqrt", 0.5, 1.0],
                "model__min_samples_leaf": [1, 2, 4],
            },
            "search": "random",
            "n_iter": 15,
        },
        "ET": {
            "pipeline": tree_pipeline(
                ExtraTreesClassifier(
                    n_estimators=300,
                    class_weight="balanced",
                    random_state=seed,
                    n_jobs=1,
                )
            ),
            "params": {
                "model__max_depth": [None, 3, 5, 8],
                "model__max_features": ["sqrt", 0.5, 1.0],
                "model__min_samples_leaf": [1, 2, 4],
            },
            "search": "random",
            "n_iter": 15,
        },
        "GBDT": {
            "pipeline": tree_pipeline(GradientBoostingClassifier(random_state=seed)),
            "params": {
                "model__n_estimators": [50, 100, 200],
                "model__learning_rate": [0.01, 0.05, 0.1],
                "model__max_depth": [1, 2, 3],
                "model__min_samples_leaf": [1, 2, 4],
                "model__subsample": [0.7, 1.0],
            },
            "search": "random",
            "n_iter": 20,
            "sample_weight": True,
        },
        "XGBoost": {
            "pipeline": tree_pipeline(
                XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=seed,
                    n_jobs=1,
                    tree_method="hist",
                )
            ),
            "params": {
                "model__n_estimators": [50, 100, 200],
                "model__learning_rate": [0.01, 0.05, 0.1],
                "model__max_depth": [1, 2, 3],
                "model__min_child_weight": [1, 3, 5],
                "model__subsample": [0.7, 1.0],
                "model__colsample_bytree": [0.7, 1.0],
                "model__reg_lambda": [1.0, 5.0, 10.0],
            },
            "search": "random",
            "n_iter": 18,
            "sample_weight": True,
            "search_n_jobs": 1,
        },
        "SVM-RBF": {
            "pipeline": scaled_pipeline(
                SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=seed)
            ),
            "params": {
                "model__C": [0.01, 0.1, 1.0, 10.0, 100.0],
                "model__gamma": ["scale", 0.001, 0.01, 0.1, 1.0],
            },
            "search": "grid",
        },
        "KNN": {
            "pipeline": scaled_pipeline(KNeighborsClassifier()),
            "params": {
                "model__n_neighbors": [3, 5, 7, 9, 11],
                "model__weights": ["uniform", "distance"],
                "model__p": [1, 2],
            },
            "search": "grid",
        },
        "sLDA": {
            "pipeline": scaled_pipeline(LinearDiscriminantAnalysis(solver="lsqr")),
            "params": {"model__shrinkage": ["auto", 0.0, 0.1, 0.3, 0.5, 0.7, 0.9]},
            "search": "grid",
        },
        "GNB": {
            "pipeline": scaled_pipeline(GaussianNB()),
            "params": {"model__var_smoothing": np.logspace(-11, -7, 7).tolist()},
            "search": "grid",
        },
    }


def load_data():
    chem = pd.read_excel(CHEM_PATH, sheet_name="Raw_2026")
    spec = pd.read_excel(SPEC_PATH, sheet_name="Spectral_features_24")
    definitions = pd.read_excel(SPEC_PATH, sheet_name="Definitions")
    ph = pd.read_excel(SPEC_PATH, sheet_name="Individual_chem_16", usecols=["Well", "pH"])
    chem = chem.merge(ph, on="Well", how="left", validate="one_to_one")

    flags = pd.DataFrame(index=chem.index)
    for feature, boundary in V_BOUNDARIES.items():
        if feature in chem.columns:
            flags[feature] = pd.to_numeric(chem[feature], errors="coerce") > boundary
    flags["pH"] = (pd.to_numeric(chem["pH"], errors="coerce") < 5.5) | (
        pd.to_numeric(chem["pH"], errors="coerce") > 9.0
    )
    chem["Risk"] = flags.any(axis=1).astype(int)
    chem["V_trigger_count"] = flags.sum(axis=1)
    chem["V_trigger_indicators"] = [", ".join(flags.columns[row.to_numpy()].tolist()) for _, row in flags.iterrows()]

    merged = spec.merge(
        chem[["Well", "Well type", "Risk", "V_trigger_count", "V_trigger_indicators"]],
        on="Well",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != 41:
        raise RuntimeError(f"Expected 41 matched wells, got {len(merged)}")

    modality_map = definitions.set_index("Variable")["Spectrum"].astype(str)
    eem = [c for c in spec.columns if c != "Well" and modality_map.get(c, "").startswith("eem")]
    uv = [c for c in spec.columns if c != "Well" and modality_map.get(c, "").startswith("UV")]
    if len(eem) != 11 or len(uv) != 13:
        raise RuntimeError(f"Unexpected modality feature counts: EEM={len(eem)}, UV={len(uv)}")
    # Direct concatenation is retained as a comparison input. It does not define
    # the study as spectral fusion or create a new spectral modality.
    feature_sets = {"EEM-only": eem, "UV-Vis-only": uv, JOINT_INPUT: eem + uv}
    return merged, flags, feature_sets


def make_search(spec, inner_cv, seed):
    common = dict(
        estimator=clone(spec["pipeline"]),
        param_distributions=spec["params"],
        scoring="average_precision",
        cv=inner_cv,
        n_jobs=spec.get("search_n_jobs", -1),
        refit=True,
        error_score="raise",
        return_train_score=False,
    )
    if spec["search"] == "random":
        return RandomizedSearchCV(
            **common,
            n_iter=min(spec.get("n_iter", 20), int(np.prod([len(v) for v in spec["params"].values()]))),
            random_state=seed,
        )
    return GridSearchCV(
        estimator=common["estimator"],
        param_grid=spec["params"],
        scoring=common["scoring"],
        cv=common["cv"],
        n_jobs=common["n_jobs"],
        refit=True,
        error_score="raise",
        return_train_score=False,
    )


def metric_row(y, prob):
    pred = (prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "ROC-AUC": roc_auc_score(y, prob),
        "PR-AUC": average_precision_score(y, prob),
        "Accuracy": accuracy_score(y, pred),
        "Balanced_accuracy": balanced_accuracy_score(y, pred),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "Specificity": tn / (tn + fp) if tn + fp else np.nan,
        "F1": f1_score(y, pred, zero_division=0),
        "MCC": matthews_corrcoef(y, pred),
        "Brier": brier_score_loss(y, prob),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def stratified_bootstrap_ci(y, prob, rng, n_boot=N_BOOT):
    y = np.asarray(y)
    prob = np.asarray(prob)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    roc_values, pr_values = [], []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos), replace=True), rng.choice(neg, len(neg), replace=True)])
        roc_values.append(roc_auc_score(y[idx], prob[idx]))
        pr_values.append(average_precision_score(y[idx], prob[idx]))
    return {
        "ROC-AUC_low": float(np.percentile(roc_values, 2.5)),
        "ROC-AUC_high": float(np.percentile(roc_values, 97.5)),
        "PR-AUC_low": float(np.percentile(pr_values, 2.5)),
        "PR-AUC_high": float(np.percentile(pr_values, 97.5)),
    }


def paired_delta_ci(y, first, second, metric, rng, n_boot=N_BOOT):
    y = np.asarray(y)
    first = np.asarray(first)
    second = np.asarray(second)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    values = []
    scorer = average_precision_score if metric == "PR-AUC" else roc_auc_score
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos), replace=True), rng.choice(neg, len(neg), replace=True)])
        values.append(scorer(y[idx], first[idx]) - scorer(y[idx], second[idx]))
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    os.environ.setdefault("PYTHONWARNINGS", "ignore::FutureWarning")
    start = time.time()
    data, flags, feature_sets = load_data()
    y = data["Risk"].to_numpy(dtype=int)
    print(f"Risk distribution: risk={y.sum()}, non-risk={(1-y).sum()}", flush=True)
    print({k: len(v) for k, v in feature_sets.items()}, flush=True)

    oof = data[["Well", "Well type", "Risk", "V_trigger_count", "V_trigger_indicators"]].copy()
    metrics = []
    inner_params = []
    loo = LeaveOneOut()

    for modality_index, modality in enumerate(MODALITY_ORDER):
        X = data[feature_sets[modality]].copy()
        for model_index, model_name in enumerate(MODEL_ORDER):
            model_start = time.time()
            prob = np.full(len(data), np.nan)
            fold_best_scores = []
            fold_params = []
            for fold, (train_idx, test_idx) in enumerate(loo.split(X, y)):
                fold_seed = SEED + modality_index * 10000 + model_index * 1000 + fold
                specs = model_specs(fold_seed)
                inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=fold_seed)
                search = make_search(specs[model_name], inner, fold_seed)
                fit_kwargs = {}
                if specs[model_name].get("sample_weight"):
                    fit_kwargs["model__sample_weight"] = compute_sample_weight(class_weight="balanced", y=y[train_idx])
                search.fit(X.iloc[train_idx], y[train_idx], **fit_kwargs)
                prob[test_idx[0]] = search.predict_proba(X.iloc[test_idx])[:, 1][0]
                fold_best_scores.append(float(search.best_score_))
                fold_params.append(search.best_params_)
                inner_params.append(
                    {
                        "Modality": modality,
                        "Model": model_name,
                        "Held_out_well": data.iloc[test_idx[0]]["Well"],
                        "Inner_best_PR_AUC": float(search.best_score_),
                        "Best_params": json.dumps(search.best_params_, ensure_ascii=False, sort_keys=True),
                    }
                )
            if np.isnan(prob).any():
                raise RuntimeError(f"Missing OOF probabilities for {modality}/{model_name}")
            oof[f"{modality}__{model_name}"] = prob
            row = {"Modality": modality, "Model": model_name, **metric_row(y, prob)}
            row["Inner_PR_AUC_mean"] = float(np.mean(fold_best_scores))
            metrics.append(row)
            pd.DataFrame(metrics).to_csv(OUT / "partial_model_metrics.csv", index=False, encoding="utf-8-sig")
            oof.to_csv(OUT / "partial_oof_probabilities.csv", index=False, encoding="utf-8-sig")
            pd.DataFrame(inner_params).to_csv(OUT / "partial_inner_cv_best_params.csv", index=False, encoding="utf-8-sig")
            elapsed = time.time() - model_start
            print(
                f"{modality:12s} {model_name:9s} PR-AUC={row['PR-AUC']:.3f} ROC-AUC={row['ROC-AUC']:.3f} "
                f"BA={row['Balanced_accuracy']:.3f} elapsed={elapsed:.1f}s",
                flush=True,
            )

    metrics_df = pd.DataFrame(metrics)
    metrics_df["Modality"] = pd.Categorical(metrics_df["Modality"], MODALITY_ORDER, ordered=True)
    metrics_df["Model"] = pd.Categorical(metrics_df["Model"], MODEL_ORDER, ordered=True)
    metrics_df = metrics_df.sort_values(["Modality", "Model"]).reset_index(drop=True)

    rng = np.random.default_rng(SEED)
    ci_rows = []
    for _, row in metrics_df.iterrows():
        col = f"{row['Modality']}__{row['Model']}"
        ci_rows.append({"Modality": str(row["Modality"]), "Model": str(row["Model"]), **stratified_bootstrap_ci(y, oof[col], rng)})
    ci_df = pd.DataFrame(ci_rows)
    metrics_df["Modality"] = metrics_df["Modality"].astype(str)
    metrics_df["Model"] = metrics_df["Model"].astype(str)
    metrics_df = metrics_df.merge(ci_df, on=["Modality", "Model"], how="left", validate="one_to_one")

    rank = metrics_df.sort_values(
        ["PR-AUC", "ROC-AUC", "Brier", "Recall"], ascending=[False, False, True, False]
    ).reset_index(drop=True)
    rank.insert(0, "Overall_rank", np.arange(1, len(rank) + 1))
    best = rank.iloc[0]
    best_modality, best_model = best["Modality"], best["Model"]
    best_col = f"{best_modality}__{best_model}"

    best_by_modality = (
        rank.sort_values(["Overall_rank"]).groupby("Modality", sort=False, as_index=False).first()
    )
    best_single = rank[rank["Modality"].isin(["EEM-only", "UV-Vis-only"])].iloc[0]
    best_joint = rank[rank["Modality"] == JOINT_INPUT].iloc[0]
    delta_summary = {}
    for comparator_name, comparator in {
        "best_EEM": rank[rank["Modality"] == "EEM-only"].iloc[0],
        "best_UV_Vis": rank[rank["Modality"] == "UV-Vis-only"].iloc[0],
    }.items():
        delta_summary[comparator_name] = {}
        for metric in ["PR-AUC", "ROC-AUC"]:
            low, high = paired_delta_ci(
                y,
                oof[f"{JOINT_INPUT}__{best_joint['Model']}"],
                oof[f"{comparator['Modality']}__{comparator['Model']}"],
                metric,
                rng,
            )
            delta_summary[comparator_name][metric] = {
                "delta_joint_input_minus_comparator": float(best_joint[metric] - comparator[metric]),
                "ci_low": low,
                "ci_high": high,
            }

    # Refit the winning pipeline on all wells with a fresh inner CV.
    final_specs = model_specs(SEED + 999999)
    final_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + 999999)
    final_search = make_search(final_specs[best_model], final_inner, SEED + 999999)
    final_fit_kwargs = {}
    if final_specs[best_model].get("sample_weight"):
        final_fit_kwargs["model__sample_weight"] = compute_sample_weight(class_weight="balanced", y=y)
    final_search.fit(data[feature_sets[best_modality]], y, **final_fit_kwargs)
    joblib.dump(
        {
            "model": final_search.best_estimator_,
            "modality": best_modality,
            "features": feature_sets[best_modality],
            "risk_boundaries": {**V_BOUNDARIES, "pH": "<5.5 or >9.0"},
            "threshold": 0.5,
        },
        MODEL_DIR / "best_model.joblib",
    )

    # Retain one optimized model for each single-spectrum input. Their probabilities
    # are passed to the LLM as parallel evidence. Direct concatenation remains a
    # model-comparison condition.
    retained_models = {}
    retained_seeds = {"EEM-only": SEED + 800000, "UV-Vis-only": SEED + 800001}
    retained_selection = {"EEM-only": "Logistic", "UV-Vis-only": "Logistic"}
    for retained_modality, retained_model_name in retained_selection.items():
        if retained_modality == best_modality and retained_model_name == best_model:
            retained_search = final_search
        else:
            retained_seed = retained_seeds[retained_modality]
            retained_specs = model_specs(retained_seed)
            retained_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=retained_seed)
            retained_search = make_search(retained_specs[retained_model_name], retained_inner, retained_seed)
            retained_fit_kwargs = {}
            if retained_specs[retained_model_name].get("sample_weight"):
                retained_fit_kwargs["model__sample_weight"] = compute_sample_weight(class_weight="balanced", y=y)
            retained_search.fit(data[feature_sets[retained_modality]], y, **retained_fit_kwargs)
        retained_models[retained_modality] = {
            "model": retained_search.best_estimator_,
            "model_name": retained_model_name,
            "features": feature_sets[retained_modality],
            "best_params": retained_search.best_params_,
            "threshold": 0.5,
        }
    joblib.dump(
        {
            "models": retained_models,
            "risk_boundaries": {**V_BOUNDARIES, "pH": "<5.5 or >9.0"},
            "llm_input": "parallel probabilities and structured evidence from all retained spectral inputs",
        },
        MODEL_DIR / "dual_spectral_models.joblib",
    )

    # Save audit-ready flat files.
    metrics_df.to_csv(OUT / "all_model_metrics.csv", index=False, encoding="utf-8-sig")
    rank.to_csv(OUT / "model_ranking.csv", index=False, encoding="utf-8-sig")
    oof.to_csv(OUT / "oof_probabilities.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(inner_params).to_csv(OUT / "inner_cv_best_params.csv", index=False, encoding="utf-8-sig")
    boundary_rows = [
        {
            "Indicator": k,
            "V_class_rule": f"> {v:g}",
            "Boundary_value": v,
            "Unit": "ug/L" if k in UG_L_INDICATORS else "mg/L",
        }
        for k, v in V_BOUNDARIES.items()
    ]
    boundary_rows.append(
        {"Indicator": "pH", "V_class_rule": "< 5.5 or > 9.0", "Boundary_value": np.nan, "Unit": "dimensionless"}
    )
    pd.DataFrame(boundary_rows).to_csv(OUT / "risk_boundaries.csv", index=False, encoding="utf-8-sig")
    well_labels = data[["Well", "Well type", "Risk", "V_trigger_count", "V_trigger_indicators"]].copy()
    well_labels.to_csv(OUT / "risk_labels.csv", index=False, encoding="utf-8-sig")

    summary = {
        "seed": SEED,
        "n_wells": int(len(data)),
        "n_risk": int(y.sum()),
        "n_nonrisk": int((1 - y).sum()),
        "feature_counts": {k: len(v) for k, v in feature_sets.items()},
        "features": feature_sets,
        "best_model": {
            "modality": best_modality,
            "model": best_model,
            "metrics": {k: (int(best[k]) if k in ["TN", "FP", "FN", "TP"] else float(best[k])) for k in [
                "ROC-AUC", "PR-AUC", "Accuracy", "Balanced_accuracy", "Precision", "Recall", "Specificity", "F1", "MCC", "Brier", "TN", "FP", "FN", "TP", "ROC-AUC_low", "ROC-AUC_high", "PR-AUC_low", "PR-AUC_high"
            ]},
            "full_data_best_params": final_search.best_params_,
            "full_data_inner_PR_AUC": float(final_search.best_score_),
        },
        "best_single": {
            "modality": best_single["Modality"],
            "model": best_single["Model"],
            "PR-AUC": float(best_single["PR-AUC"]),
            "ROC-AUC": float(best_single["ROC-AUC"]),
        },
        "best_by_modality": best_by_modality.to_dict(orient="records"),
        "retained_model_outputs": {
            modality: {
                "model": item["model_name"],
                "features": item["features"],
                "best_params": item["best_params"],
                "threshold": item["threshold"],
            }
            for modality, item in retained_models.items()
        },
        "paired_delta_joint_input": delta_summary,
        "runtime_seconds": time.time() - start,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("BEST", json.dumps(summary["best_model"], ensure_ascii=False), flush=True)
    print("Runtime seconds", summary["runtime_seconds"], flush=True)


if __name__ == "__main__":
    main()
