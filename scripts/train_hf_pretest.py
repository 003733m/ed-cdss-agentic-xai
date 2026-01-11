#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import shap

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance

from cdss.features import build_feature_groups


def _to_num(s):
    return pd.to_numeric(s, errors="coerce")


def _make_numeric_and_impute(X: pd.DataFrame) -> pd.DataFrame:
    for c in X.columns:
        X[c] = _to_num(X[c])
    X = X.dropna(axis=1, how="all")
    X = X.apply(lambda col: col.fillna(col.median()) if col.notna().any() else col)
    return X


def build_hf_label_mcq160b(df: pd.DataFrame):
    """
    MCQ160B: congestive heart failure ever told
    codebook: 1 Yes, 2 No, 7 Refused, 9 Don't know, missing
    We keep only 1/2.
    """
    if "MCQ160B" not in df.columns:
        raise KeyError("MCQ160B missing from dataframe (HF label).")
    raw = _to_num(df["MCQ160B"])
    mask = raw.isin([1, 2])
    y = (raw[mask] == 1).astype(int)
    return y, mask


def train_hf(df: pd.DataFrame, artifacts_dir: str = "artifacts", seed: int = 42) -> dict:
    """
    HF PRETEST (Strict leakage-free):
    - Label uses MCQ160B (history)
    - Features exclude MCQ/BPQ/CDQ/RX/BPX and label col
    - Uses demographics + anthropometry + questionnaire_lifestyle only (same as your notebook policy)
    """
    os.makedirs(artifacts_dir, exist_ok=True)

    feature_groups, survey_vars = build_feature_groups(df)

    # Label + mask
    y, mask = build_hf_label_mcq160b(df)

    allowed_groups = ["demographics", "anthropometry", "questionnaire_lifestyle"]
    X_cols = []
    for g in allowed_groups:
        X_cols += (feature_groups.get(g, []) or [])
    X_cols = list(dict.fromkeys([c for c in X_cols if c in df.columns]))

    LABEL_COL = "MCQ160B"

    def is_forbidden(c: str) -> bool:
        u = str(c).strip().upper()
        return (
            u.startswith("MCQ")
            or u.startswith("BPQ")
            or u.startswith("CDQ")
            or u.startswith("RX")
            or u.startswith("BPX")
            or u == LABEL_COL
        )

    X_cols = [c for c in X_cols if not is_forbidden(c)]

    # HARD GUARANTEE
    for bad_prefix in ["MCQ", "BPQ", "CDQ", "RX", "BPX"]:
        assert all(not str(c).strip().upper().startswith(bad_prefix) for c in X_cols), f"Leakage: {bad_prefix} in X!"
    assert LABEL_COL not in X_cols, "Leakage: label col in X!"

    X = df.loc[mask, X_cols].copy()
    X = _make_numeric_and_impute(X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
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

    pred = model.predict_proba(X_test)[:, 1]
    auc = float(roc_auc_score(y_test, pred))

    # Permutation importance
    perm = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=5,
        random_state=seed,
        scoring="roc_auc",
        n_jobs=1,
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

    # ---- Save artifacts (STANDARD NAMES) ----
    joblib.dump(model, os.path.join(artifacts_dir, "hf_model.joblib"))
    joblib.dump(explainer, os.path.join(artifacts_dir, "hf_explainer.joblib"))

    with open(os.path.join(artifacts_dir, "hf_feature_cols.json"), "w", encoding="utf-8") as f:
        json.dump(list(X.columns), f, indent=2)

    with open(os.path.join(artifacts_dir, "hf_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "auc": auc,
                "label_col": LABEL_COL,
                "n_samples": int(len(y)),
                "n_features": int(X.shape[1]),
            },
            f,
            indent=2,
        )

    meta = {
        "task": "heart_failure_by_mcq",
        "label_col": LABEL_COL,
        "label_policy": "keep only 1/2; (1=yes -> 1, 2=no -> 0)",
        "policy": {
            "allowed_groups": allowed_groups,
            "forbidden_prefixes": ["MCQ", "BPQ", "CDQ", "RX", "BPX"],
            "label_excluded": True,
            "survey_vars_excluded": survey_vars,
        },
        "feature_cols": list(X.columns),
        "auc": auc,
        "top_features_perm": perm_top20,
        "top_features_shap": shap_top20,
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
    print("✅ Saved artifacts to:", args.artifacts_dir)
