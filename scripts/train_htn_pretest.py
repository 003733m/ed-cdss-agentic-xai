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
    ConfusionMatrixDisplay,
)
from sklearn.inspection import permutation_importance

from cdss.features import build_feature_groups

plt.switch_backend("Agg")


def _to_num(s):
    return pd.to_numeric(s, errors="coerce")


def _mean_cols(df_, cols):
    cols = [c for c in cols if c in df_.columns]
    if not cols:
        return pd.Series(np.nan, index=df_.index)
    return df_[cols].apply(_to_num).mean(axis=1, skipna=True)


def build_hypertension_by_bpx(df_: pd.DataFrame) -> pd.DataFrame:
    """Label = HTN by BPX measurement means (>=130/80). Also returns stage."""
    sbp_cols = [c for c in df_.columns if re.match(r"^BPXSY[1-4]$", str(c))]
    dbp_cols = [c for c in df_.columns if re.match(r"^BPXDI[1-4]$", str(c))]

    sbp_mean = _mean_cols(df_, sbp_cols)
    dbp_mean = _mean_cols(df_, dbp_cols)

    valid = sbp_mean.notna() | dbp_mean.notna()

    htn = pd.Series(np.nan, index=df_.index, dtype="float")
    htn.loc[valid] = ((sbp_mean.loc[valid] >= 130) | (dbp_mean.loc[valid] >= 80)).astype(float)

    stage = pd.Series(np.nan, index=df_.index, dtype="float")
    s2 = (sbp_mean >= 140) | (dbp_mean >= 90)
    s1 = (((sbp_mean >= 130) & (sbp_mean < 140)) | ((dbp_mean >= 80) & (dbp_mean < 90))) & (~s2)
    s0 = (~s1) & (~s2) & valid

    stage.loc[s0] = 0.0
    stage.loc[s1] = 1.0
    stage.loc[s2] = 2.0

    return pd.DataFrame(
        {"hypertension_by_bp": htn, "sbp_mean": sbp_mean, "dbp_mean": dbp_mean, "htn_stage": stage}
    )


def is_bpx_or_bpq(c: str) -> bool:
    u = str(c).strip().upper()
    return u.startswith("BPX") or u.startswith("BPQ")


def _make_numeric_and_impute(X: pd.DataFrame) -> pd.DataFrame:
    for c in X.columns:
        X[c] = _to_num(X[c])
    X = X.dropna(axis=1, how="all")
    X = X.apply(lambda col: col.fillna(col.median()) if col.notna().any() else col)
    return X


def train_htn(df: pd.DataFrame, artifacts_dir: str = "artifacts", seed: int = 42) -> dict:
    """
    HTN PRETEST (Leakage-free, registry.py compatible):
    - Label uses BPX measurements (>=130/80)
    - Features exclude BPX & BPQ
    - Uses demographics + anthropometry + questionnaire_lifestyle
    - Outputs (artifacts_dir):
        htn_model.joblib
        htn_explainer.joblib
        htn_artifact.json
        htn_metrics.json
        htn_feature_cols.json
        htn_confusion_matrix.png (optional)
    """
    os.makedirs(artifacts_dir, exist_ok=True)

    feature_groups, survey_vars = build_feature_groups(df)

    # ---- Label ----
    htn_df = build_hypertension_by_bpx(df)
    y = htn_df["hypertension_by_bp"]
    valid = y.notna()
    yv = y.loc[valid].astype(int)

    # ---- Features (leakage-free) ----
    allowed_groups = ["demographics", "anthropometry", "questionnaire_lifestyle"]
    X_cols = []
    for g in allowed_groups:
        X_cols += (feature_groups.get(g, []) or [])
    X_cols = list(dict.fromkeys([c for c in X_cols if c in df.columns]))
    X_cols = [c for c in X_cols if not is_bpx_or_bpq(c)]

    X = df.loc[valid, X_cols].copy()
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
    disp = ConfusionMatrixDisplay(confusion_matrix=total_cm, display_labels=["No HTN", "HTN"])
    disp.plot(cmap="Greens", values_format="d", ax=ax)
    plt.title("Cumulative Confusion Matrix (5-Fold Sum) - HTN")
    cm_path = os.path.join(artifacts_dir, "htn_confusion_matrix.png")
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
    joblib.dump(model, os.path.join(artifacts_dir, "htn_model.joblib"))
    joblib.dump(explainer, os.path.join(artifacts_dir, "htn_explainer.joblib"))

    with open(os.path.join(artifacts_dir, "htn_feature_cols.json"), "w", encoding="utf-8") as f:
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
    with open(os.path.join(artifacts_dir, "htn_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)

    meta = {
        "task": "hypertension_by_bp",
        "label": "BPX-based (>=130/80)",
        "policy": {
            "allowed_groups": allowed_groups,
            "forbidden_prefixes": ["BPX", "BPQ"],
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
    with open(os.path.join(artifacts_dir, "htn_artifact.json"), "w", encoding="utf-8") as f:
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
    meta = train_htn(df, artifacts_dir=args.artifacts_dir, seed=args.seed)
    print("✅ HTN trained | AUC:", meta["auc"])
    print("✅ Saved artifacts to:", args.artifacts_dir)
