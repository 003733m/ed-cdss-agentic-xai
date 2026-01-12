#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
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
    ConfusionMatrixDisplay
)
from sklearn.inspection import permutation_importance

from cdss.features import build_feature_groups

# Server environments usually don't have a screen
plt.switch_backend('Agg')

# ----------------------------
# Helpers
# ----------------------------
def _to_num(s):
    return pd.to_numeric(s, errors="coerce")

def _make_numeric_and_impute(X: pd.DataFrame) -> pd.DataFrame:
    # numeric coerce
    for c in X.columns:
        X[c] = _to_num(X[c])
    # drop all NaN columns
    X = X.dropna(axis=1, how="all")
    # median impute
    X = X.apply(lambda col: col.fillna(col.median()) if col.notna().any() else col)
    return X

# ----------------------------
# Train HF
# ----------------------------
def train_hf(df: pd.DataFrame, artifacts_dir: str = "artifacts", seed: int = 42) -> dict:
    """
    HF PRETEST (Leakage-free):
    - Label: MCQ160B (Ever told had congestive heart failure)
    - Features: Excludes MCQ, BPQ, CDQ, RX, BPX (Strict Mode)
    - Uses: Demographics + Anthropometry + Lifestyle
    - Includes: 5-Fold CV, Green Confusion Matrix, Detailed Metrics
    """
    os.makedirs(artifacts_dir, exist_ok=True)

    feature_groups, survey_vars = build_feature_groups(df)

    # 1. Label Logic (MCQ160B)
    label_col = "MCQ160B"
    if label_col not in df.columns:
        raise ValueError(f"Label column {label_col} not found in dataframe.")
    
    raw = pd.to_numeric(df[label_col], errors="coerce")
    # NHANES: 1=Yes, 2=No. Others (7,9) are missing/refused.
    mask = raw.isin([1, 2])
    
    # Define y (Target)
    y_full = (raw[mask] == 1).astype(int)
    
    # 2. Leakage-free Features (Strict)
    allowed_groups = ["demographics", "anthropometry", "questionnaire_lifestyle"]
    X_cols = []
    for g in allowed_groups:
        X_cols += (feature_groups.get(g, []) or [])
    
    # Remove duplicates and ensure existence
    X_cols = list(dict.fromkeys([c for c in X_cols if c in df.columns]))
    
    # Forbidden Prefixes for HF (Strict Pretest)
    forbidden_prefixes = ["MCQ", "BPQ", "CDQ", "RX", "BPX"]
    
    def is_forbidden(c):
        u = str(c).strip().upper()
        return any(u.startswith(p) for p in forbidden_prefixes) or (u == label_col)

    X_cols = [c for c in X_cols if not is_forbidden(c)]

    # 3. Build Modeling Table
    X = df.loc[mask, X_cols].copy()
    yv = y_full  # Valid targets

    X = _make_numeric_and_impute(X)

    # 4. Main Split (for Final Model & SHAP)
    X_train, X_test, y_train, y_test = train_test_split(
        X, yv, test_size=0.2, stratify=yv, random_state=seed
    )

    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    spw = (neg / pos) if pos > 0 else 1.0

    # Train Main Model
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

    pred = model.predict_proba(X_test)[:, 1]
    auc = float(roc_auc_score(y_test, pred))

    # ---------------------------------------------------------
    # 5-Fold Cross Validation (Robust Metrics)
    # ---------------------------------------------------------
    print("⏳ Running 5-Fold Cross Validation (HF)...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    
    cv_metrics = {"acc": [], "f1": [], "rec": [], "prec": [], "auc": []}
    total_cm = np.zeros((2, 2), dtype=int)

    for fold_idx, (t_idx, v_idx) in enumerate(skf.split(X, yv)):
        X_t, X_v = X.iloc[t_idx], X.iloc[v_idx]
        y_t, y_v_fold = yv.iloc[t_idx], yv.iloc[v_idx]

        # Dynamic SPW per fold
        pos_n = y_t.sum()
        neg_n = len(y_t) - pos_n
        spw_cv = (neg_n / pos_n) if pos_n > 0 else 1.0

        cv_model = xgb.XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, scale_pos_weight=spw_cv,
            eval_metric="logloss", random_state=seed, n_jobs=-1, verbosity=0
        )
        cv_model.fit(X_t, y_t)
        
        preds_cv = cv_model.predict(X_v)
        proba_cv = cv_model.predict_proba(X_v)[:, 1]

        cv_metrics["acc"].append(accuracy_score(y_v_fold, preds_cv))
        cv_metrics["f1"].append(f1_score(y_v_fold, preds_cv))
        cv_metrics["rec"].append(recall_score(y_v_fold, preds_cv))
        cv_metrics["prec"].append(precision_score(y_v_fold, preds_cv))
        cv_metrics["auc"].append(roc_auc_score(y_v_fold, proba_cv))
        
        total_cm += confusion_matrix(y_v_fold, preds_cv)

    # Plot & Save Cumulative Confusion Matrix (Green)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=total_cm, display_labels=["No HF", "HF"])
    disp.plot(cmap="Greens", values_format="d", ax=ax)
    plt.title("Cumulative Confusion Matrix (5-Fold Sum) - HF")
    cm_path = os.path.join(artifacts_dir, "hf_confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"✅ Saved Confusion Matrix to {cm_path}")

    # ---------------------------------------------------------
    # Feature Importance
    # ---------------------------------------------------------
    # Permutation (Global)
    perm = permutation_importance(
        model, X_test, y_test, n_repeats=5, random_state=seed, scoring="roc_auc", n_jobs=1
    )
    perm_imp = pd.Series(perm.importances_mean, index=X_test.columns).sort_values(ascending=False)
    perm_top20 = perm_imp.head(20).index.tolist()

    # SHAP
    explainer = shap.TreeExplainer(model)
    X_shap = X_test.sample(n=min(500, len(X_test)), random_state=seed)
    sv = explainer.shap_values(X_shap)
    sv_pos = sv[1] if isinstance(sv, list) else sv
    shap_imp = pd.Series(np.abs(sv_pos).mean(axis=0), index=X_shap.columns).sort_values(ascending=False)
    shap_top20 = shap_imp.head(20).index.tolist()

    # ---- Save Artifacts ----
    joblib.dump(model, os.path.join(artifacts_dir, "hf_model.joblib"))
    joblib.dump(explainer, os.path.join(artifacts_dir, "hf_explainer.joblib"))

    with open(os.path.join(artifacts_dir, "hf_feature_cols.json"), "w", encoding="utf-8") as f:
        json.dump(list(X.columns), f, indent=2)

    # Detailed Metrics JSON
    metrics_data = {
        "auc": auc,
        "n_samples": int(len(yv)),
        "n_features": int(X.shape[1]),
        "cv_mean_auc": float(np.mean(cv_metrics["auc"])),
        "cv_mean_acc": float(np.mean(cv_metrics["acc"])),
        "cv_mean_f1": float(np.mean(cv_metrics["f1"])),
        "cv_mean_recall": float(np.mean(cv_metrics["rec"])),
        "cv_mean_precision": float(np.mean(cv_metrics["prec"]))
    }
    with open(os.path.join(artifacts_dir, "hf_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)

    meta = {
        "task": "heart_failure_by_mcq",
        "label": "MCQ160B",
        "policy": {
            "allowed_groups": allowed_groups,
            "forbidden_prefixes": forbidden_prefixes,
            "survey_vars_excluded": survey_vars,
        },
        "feature_cols": list(X.columns),
        "auc": auc,
        "top_features_perm": perm_top20,
        "top_features_shap": shap_top20,
        "cv_metrics": {
            "mean_auc": float(np.mean(cv_metrics["auc"])),
            "mean_f1": float(np.mean(cv_metrics["f1"])),
            "mean_recall": float(np.mean(cv_metrics["rec"])),
            "cumulative_cm": total_cm.tolist()
        },
        "shap_available": True,
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
    print(f"✅ CV Mean Recall: {meta['cv_metrics']['mean_recall']:.4f}")
    print("✅ Saved artifacts to:", args.artifacts_dir)
