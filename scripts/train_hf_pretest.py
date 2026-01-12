#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.inspection import permutation_importance

from cdss.features import build_feature_groups

plt.switch_backend("Agg")


def _to_num(s):
    return pd.to_numeric(s, errors="coerce")


def _make_numeric_and_impute(X: pd.DataFrame) -> pd.DataFrame:
    for c in X.columns:
        X[c] = _to_num(X[c])
    X = X.dropna(axis=1, how="all")
    X = X.apply(lambda col: col.fillna(col.median()) if col.notna().any() else col)
    return X


def train_hf(df: pd.DataFrame, artifacts_dir: str = "artifacts", seed: int = 42) -> dict:
    """
    HF PRETEST (Leakage-free, registry.py compatible):
    - Label: MCQ160B (1=yes, 2=no; keep only 1/2)
    - Features: demographics + anthropometry + questionnaire_lifestyle
    - Forbidden prefixes: MCQ/BPQ/CDQ/RX/BPX + label col
    - Outputs (artifacts_dir):
        hf_model.joblib
        hf_explainer.joblib
        hf_artifact.json
        hf_metrics.json
        hf_feature_cols.json
        hf_confusion_matrix.png (optional)
    """
    os.makedirs(artifacts_dir, exist_ok=True)

    feature_groups, survey_vars = build_feature_groups(df)

    # ---- Label ----
    label_col = "MCQ160B"
    if label_col not in df.columns:
        raise KeyError(f"HF label column missing: {label_col}")

    raw = _to_num(df[label_col])
    mask = raw.isin([1, 2])
    yv = (raw[mask] == 1).astype(int)

    # ---- Features (strict) ----
    allowed_groups = ["demographics", "anthropometry", "questionnaire_lifestyle"]
    X_cols = []
    for g in allowed_groups:
        X_cols += (feature_groups.get(g, []) or [])
    X_cols = list(dict.fromkeys([c for c in X_cols if c in df.columns]))

    forbidden_prefixes = ["MCQ", "BPQ", "CDQ", "RX", "BPX"]  # DIQ intentionally not included per your note

    def is_forbidden(c: str) -> bool:
        u = str(c).strip().upper()
        return any(u.startswith(p) for p in forbidden_prefixes) or (u == label_col.upper())

    X_cols = [c for c in X_cols if not is_forbidden(c)]

    # ---- Modeling table ----
    X = df.loc[mask, X_cols].copy()
    X = _make_numeric_and_impute(X)

    # ---- Train/test split (final model) ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, yv, test_size=0.2, stratify=yv, random_state=seed
    )

    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    spw = (neg / pos) if pos > 0 else 1.0

    model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=spw,
        eval_metric="logloss",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    auc = float(roc_auc_score(y_test, proba))

    # ---- 5-fold CV metrics + cumulative CM ----
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    cv_metrics = {"acc": [], "f1": [], "rec": [], "prec": [], "auc": []}
    total_cm = np.zeros((2, 2), dtype=int)

    for t_idx, v_idx in skf.split(X, yv):
        X_t, X_v = X.iloc[t_idx], X.iloc[v_idx]
        y_t, y_v_fold = yv.iloc[t_idx], yv.iloc[v_idx]

        pos_n = int(y_t.sum())
        neg_n = int(len(y_t) - pos_n)
        spw_cv = (neg_n / pos_n) if pos_n > 0 else 1.0

        cv_model = xgb.XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=spw_cv,
            eval_metric="logloss",
            random_state=seed,
            n_jobs=-1,
            verbosity=0,
        )
        cv_model.fit(X_t, y_t)

        preds = cv_model.predict(X_v)
        probv = cv_model.predict_proba(X_v)[:, 1]

        cv_metrics["acc"].append(accuracy_score(y_v_fold, preds))
        cv_metrics["f1"].append(f1_score(y_v_fold, preds))
        cv_metrics["rec"].append(recall_score(y_v_fold, preds))
        cv_metrics["prec"].append(precision_score(y_v_fold, preds))
        cv_metrics["auc"].append(roc_auc_score(y_v_fold, probv))

        total_cm += confusion_matrix(y_v_fold, preds)

    # save CM plot
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=total_cm, display_labels=["No HF", "HF"])
    disp.plot(cmap="Greens", values_format="d", ax=ax)
    plt.title("Cumulative Confusion Matrix (5-Fold Sum) - HF")
    cm_path = os.path.join(artifacts_dir, "hf_confusion_matrix.png")
    plt.savefig(cm_path, bbox_inches="tight")
    plt.close()

    # ---- Global importance (perm) ----
    perm = permutation_importance(
        model, X_test, y_test,
        n_repeats=5, random_state=seed, scoring="roc_auc", n_jobs=1
    )
    perm_imp = pd.Series(perm.importances_mean, index=X_test.columns).sort_values(ascending=False)
    perm_top20 = perm_imp.head(20).index.tolist()

    # ---- SHAP ----
    explainer = shap.TreeExplainer(model)
    X_shap = X_test.sample(n=min(500, len(X_test)), random_state=seed)
    sv = explainer.shap_values(X_shap)
    sv_pos = sv[1] if isinstance(sv, list) else sv
    shap_imp = pd.Series(np.abs(sv_pos).mean(axis=0), index=X_shap.columns).sort_values(ascending=False)
    shap_top20 = shap_imp.head(20).index.tolist()

    # ---- Save (registry-compatible names) ----
    joblib.dump(model, os.path.join(artifacts_dir, "hf_model.joblib"))
    joblib.dump(explainer, os.path.join(artifacts_dir, "hf_explainer.joblib"))

    with open(os.path.join(artifacts_dir, "hf_feature_cols.json"), "w", encoding="utf-8") as f:
        json.dump(list(X.columns), f, indent=2)

    metrics_data = {
        "auc": auc,
        "n_samples": int(len(yv)),
        "n_features": int(X.shape[1]),
        "cv_mean_auc": float(np.mean(cv_metrics["auc"])),
        "cv_mean_acc": float(np.mean(cv_metrics["acc"])),
        "cv_mean_f1": float(np.mean(cv_metrics["f1"])),
        "cv_mean_recall": float(np.mean(cv_metrics["rec"])),
        "cv_mean_precision": float(np.mean(cv_metrics["prec"])),
        "cumulative_cm_5fold_sum": total_cm.tolist(),
    }
    with open(os.path.join(artifacts_dir, "hf_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)

    meta = {
        "task": "heart_failure_by_mcq",
        "label_col": label_col,
        "label_policy": "keep only 1/2; (1=yes -> 1, 2=no -> 0)",
        "policy": {
            "allowed_groups": allowed_groups,
            "forbidden_prefixes": forbidden_prefixes,
            "label_excluded": True,
            "survey_vars_excluded": survey_vars,
        },
        "feature_cols": list(X.columns),
        "auc": auc,
        "top_features_perm": perm_top20,
        "top_features_shap": shap_top20,
        "shap_available": True,
        "cv_metrics": {
            "mean_auc": float(np.mean(cv_metrics["auc"])),
            "mean_f1": float(np.mean(cv_metrics["f1"])),
            "mean_recall": float(np.mean(cv_metrics["rec"])),
        },
    }
    with open(os.path.join(artifacts_dir, "hf_artifact.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return meta


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="NHANES_Merged_Data.csv path")
    ap.add_argument("--artifacts-dir", default="artifacts")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.csv, low_memory=False)
    meta = train_hf(df, artifacts_dir=args.artifacts_dir, seed=args.seed)
    print("✅ HF trained | AUC:", meta["auc"])
    print("✅ Saved artifacts to:", args.artifacts_dir)
