from __future__ import annotations

from pathlib import Path
import warnings
from math import comb

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import LeaveOneOut, StratifiedKFold

from train_nested_cv import SEED, load_data, make_search, model_specs


ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "analysis" / "machine_learning" / "results"
MODEL_DIR = ROOT / "models" / "machine_learning"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SELECTED = {"EEM-only": "eem", "UV-Vis-only": "uv_vis"}

EVIDENCE_MAP = {
    "F_integral": "总体荧光响应，结合耗氧量和综合化学异常核查",
    "C_peak": "长波腐殖质样荧光，结合耗氧量核查",
    "T_peak": "短波蛋白样荧光，结合有机物组成核查",
    "C/T": "长波与短波荧光的相对贡献",
    "C_Ex": "长波荧光峰激发位置",
    "C_Em": "长波荧光峰发射位置",
    "T_Ex": "短波荧光峰激发位置",
    "T_Em": "短波荧光峰发射位置",
    "FI": "荧光来源指数，用于溶解性有机质组成研判",
    "BIX": "生物源指数，用于溶解性有机质组成研判",
    "HIX": "腐殖化指数，用于溶解性有机质组成研判",
    "A200": "短波紫外吸收，结合耗氧量、TDS和综合化学异常核查",
    "A220": "短波紫外吸收，结合耗氧量、TDS和综合化学异常核查",
    "A254": "短波紫外吸收，优先核查耗氧量及综合化学异常",
    "A280": "短波紫外吸收，结合耗氧量及综合化学异常核查",
    "A320": "中波紫外吸收带，核查吸收带延伸和有机物组成",
    "A364": "中波紫外吸收带，核查吸收带延伸和有机物组成",
    "A400": "紫外与可见区过渡响应",
    "A496": "可见显色响应，优先核查pH和显色基质效应",
    "A578": "可见显色响应，优先核查pH和显色基质效应",
    "MeanA_200_250": "短波紫外平均吸收，结合耗氧量和TDS核查",
    "MeanA_240_360": "中波紫外平均吸收，结合耗氧量和有机物组成核查",
    "Amax_190_780": "全谱最大吸收，结合TDS及短波异常核查",
    "VisMax_400_700": "可见区最大响应，优先核查pH和显色基质效应",
}

def exact_probability_shap(coefficients, intercept, test_z, background):
    """Exact reference-based SHAP values for a logistic probability output."""
    n_features = len(coefficients)
    coalition_ids = np.arange(2**n_features, dtype=np.uint32)
    masks = ((coalition_ids[:, None] >> np.arange(n_features)) & 1).astype(float)
    delta = test_z - background
    coalition_z = background + masks * delta
    logits = intercept + coalition_z @ coefficients
    values = 1.0 / (1.0 + np.exp(-np.clip(logits, -700, 700)))
    shap_values = np.zeros(n_features, dtype=float)
    subset_sizes = masks.sum(axis=1).astype(int)
    for feature_index in range(n_features):
        absent = masks[:, feature_index] == 0
        without_ids = coalition_ids[absent]
        with_ids = without_ids | (1 << feature_index)
        sizes = subset_sizes[absent]
        weights = np.array([1.0 / (n_features * comb(n_features - 1, int(size))) for size in sizes])
        shap_values[feature_index] = np.sum(weights * (values[with_ids] - values[without_ids]))
    return shap_values, float(values[0]), float(values[-1])


def compute_oof_linear_shap():
    data, _, feature_sets = load_data()
    y = data["Risk"].to_numpy(int)
    loo = LeaveOneOut()
    existing = pd.read_csv(RESULTS / "oof_probabilities.csv")
    shap_rows = []
    probability_rows = []

    for modality_index, (modality, display) in enumerate(SELECTED.items()):
        features = feature_sets[modality]
        X = data[features].copy()
        probabilities = np.full(len(data), np.nan)
        for fold, (train_idx, test_idx) in enumerate(loo.split(X, y)):
            fold_seed = SEED + modality_index * 10000 + fold
            spec = model_specs(fold_seed)["Logistic"]
            inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=fold_seed)
            search = make_search(spec, inner, fold_seed)
            search.fit(X.iloc[train_idx], y[train_idx])
            pipeline = search.best_estimator_
            probabilities[test_idx[0]] = pipeline.predict_proba(X.iloc[test_idx])[:, 1][0]

            imputer = pipeline.named_steps["imputer"]
            scaler = pipeline.named_steps["scaler"]
            model = pipeline.named_steps["model"]
            train_imputed = imputer.transform(X.iloc[train_idx])
            test_imputed = imputer.transform(X.iloc[test_idx])
            train_z = scaler.transform(train_imputed)
            test_z = scaler.transform(test_imputed)[0]
            background = train_z.mean(axis=0)
            coefficients = model.coef_[0]
            shap_values, base_value, decision_value = exact_probability_shap(
                coefficients, float(model.intercept_[0]), test_z, background
            )
            if not np.isclose(base_value + shap_values.sum(), decision_value, atol=1e-9):
                raise RuntimeError("Linear SHAP additivity check failed")

            well = data.iloc[test_idx[0]]["Well"]
            for feature, raw_value, z_value, coefficient, shap_value in zip(features, X.iloc[test_idx[0]], test_z, coefficients, shap_values):
                shap_rows.append(
                    {
                        "Well": well,
                        "Risk": int(y[test_idx[0]]),
                        "Modality": display,
                        "Feature": feature,
                        "Feature_value": float(raw_value) if pd.notna(raw_value) else np.nan,
                        "Standardized_value": float(z_value),
                        "Fold_coefficient": float(coefficient),
                        "SHAP_probability": float(shap_value),
                        "Absolute_SHAP": float(abs(shap_value)),
                        "Evidence_mapping": EVIDENCE_MAP.get(feature, ""),
                    }
                )
            probability_rows.append(
                {
                    "Well": well,
                    "Risk": int(y[test_idx[0]]),
                    "Modality": display,
                    "OOF_probability": probabilities[test_idx[0]],
                    "SHAP_base_probability": base_value,
                    "SHAP_sum_probability": float(shap_values.sum()),
                    "Predicted_probability_from_SHAP": decision_value,
                    "Best_params": str(search.best_params_),
                }
            )

        reference = existing[f"{modality}__Logistic"].to_numpy(float)
        difference = np.max(np.abs(reference - probabilities))
        if difference > 1e-10:
            raise RuntimeError(f"OOF probability mismatch for {modality}: {difference}")

    return data, pd.DataFrame(shap_rows), pd.DataFrame(probability_rows)


def build_llm_evidence(data, shap_long, probabilities):
    output = data[["Well", "Well type", "Risk", "V_trigger_count", "V_trigger_indicators"]].copy()
    prob_wide = probabilities.pivot(index="Well", columns="Modality", values="OOF_probability").reset_index()
    prob_wide = prob_wide.rename(columns={"eem": "EEM_risk_probability", "uv_vis": "UV-Vis_risk_probability"})
    output = output.merge(prob_wide, on="Well", validate="one_to_one")
    output["Model_agreement"] = np.where(
        (output["EEM_risk_probability"] >= 0.5) == (output["UV-Vis_risk_probability"] >= 0.5),
        "Agreement",
        "Divergence",
    )
    output["EEM_prediction"] = (output["EEM_risk_probability"] >= 0.5).astype(int)
    output["UV-Vis_prediction"] = (output["UV-Vis_risk_probability"] >= 0.5).astype(int)
    output["Spectral_evidence_state"] = np.select(
        [
            (output["EEM_prediction"] == 1) & (output["UV-Vis_prediction"] == 1),
            (output["EEM_prediction"] == 1) & (output["UV-Vis_prediction"] == 0),
            (output["EEM_prediction"] == 0) & (output["UV-Vis_prediction"] == 1),
        ],
        ["Both high", "EEM high", "UV-Vis high"],
        default="Both low",
    )

    for modality in ["eem", "uv_vis"]:
        subset = shap_long[shap_long["Modality"] == modality]
        positive = subset[subset["SHAP_probability"] > 0].sort_values(["Well", "SHAP_probability"], ascending=[True, False])
        negative = subset[subset["SHAP_probability"] < 0].sort_values(["Well", "SHAP_probability"], ascending=[True, True])
        for direction, frame in [("positive", positive), ("negative", negative)]:
            top = frame.groupby("Well", sort=False).head(3)
            features = top.groupby("Well").apply(
                lambda x: "; ".join(f"{row.Feature} ({row.SHAP_probability:+.3f})" for row in x.itertuples()),
                include_groups=False,
            )
            mapping = top.groupby("Well").apply(
                lambda x: "; ".join(dict.fromkeys(x["Evidence_mapping"].tolist())),
                include_groups=False,
            )
            output = output.merge(
                features.rename(f"{modality}_top_{direction}_SHAP").reset_index(), on="Well", how="left", validate="one_to_one"
            )
            output = output.merge(
                mapping.rename(f"{modality}_{direction}_evidence_mapping").reset_index(), on="Well", how="left", validate="one_to_one"
            )
    text_columns = [column for column in output.columns if "SHAP" in column or "evidence_mapping" in column]
    output[text_columns] = output[text_columns].fillna("No contribution in this direction")
    return output


def main():
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    data, shap_long, probabilities = compute_oof_linear_shap()
    importance = shap_long.groupby(["Modality", "Feature"], as_index=False).agg(
        Mean_absolute_SHAP=("Absolute_SHAP", "mean"),
        Median_absolute_SHAP=("Absolute_SHAP", "median"),
        Mean_SHAP=("SHAP_probability", "mean"),
        Positive_coefficient_fraction=("Fold_coefficient", lambda x: float(np.mean(x > 0))),
    )
    importance["Coefficient_sign_stability"] = np.maximum(
        importance["Positive_coefficient_fraction"], 1 - importance["Positive_coefficient_fraction"]
    )
    importance = importance.sort_values(["Modality", "Mean_absolute_SHAP"], ascending=[True, False])
    llm_evidence = build_llm_evidence(data, shap_long, probabilities)

    shap_long.to_csv(RESULTS / "oof_shap_values_long.csv", index=False, encoding="utf-8-sig")
    probabilities.to_csv(RESULTS / "oof_shap_additivity_check.csv", index=False, encoding="utf-8-sig")
    importance.to_csv(RESULTS / "shap_global_importance.csv", index=False, encoding="utf-8-sig")
    llm_evidence.to_csv(RESULTS / "llm_model_evidence.csv", index=False, encoding="utf-8-sig")
    deployment_bundle = joblib.load(MODEL_DIR / "dual_spectral_models.joblib")
    deployment_bundle["shap_explanation"] = {
        "output_scale": "risk probability",
        "background": "training-set mean after fold-specific imputation and standardization",
        "method": "exact reference-based Shapley values for Logistic probability output",
        "evidence_mapping": EVIDENCE_MAP,
    }
    joblib.dump(deployment_bundle, MODEL_DIR / "dual_spectral_models_with_shap.joblib")

    print(importance.groupby("Modality").head(6).to_string(index=False))


if __name__ == "__main__":
    main()
