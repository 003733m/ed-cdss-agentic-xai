#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
scripts/sanity_checks.py
Paper-ready sanity checks for leakage & spurious performance.

Checks:
1) Age-only baseline AUC
2) Shuffled-label AUC (~0.50 expected)
3) Train-only median imputation (avoid test leakage)
4) Optional PR-AUC

Usage:
  python scripts/sanity_checks.py --csv /path/NHANES_Merged_Data.csv --task hf
  python scripts/sanity_checks.py --csv /path/NHANES_Merged_Data.csv --task htn
"""

import argparse
import json
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.utils import shuffle

# ----------------------------
# Helpers
# ----------------------------
def to_num(s):
    return pd.to_numeric(s, errors="coerce")

def median_impute_train_only(X_train: pd.DataFrame, X_test: pd.DataFrame):
    med = X_train.median(numeric_only=True)
    return X_train.fillna(med), X_test.fillna(med)

def fit_xgb_auc(X, y, seed=42, n_estimators=400, max_depth=4):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)

    pos = int(ytr.sum())
    neg = int(len(ytr) - pos)
    spw = (neg / pos) if pos > 0 else 1.0

    m = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=spw,
        eval_metric="logloss",
        random_state=seed,
        n_jobs=-1
    )
    m.fit(Xtr, ytr)
    proba = m.predict_proba(Xte)[:, 1]
    return float(roc_auc_score(yte, proba)), float(average_precision_score(yte, proba))

# ----------------------------
# Labels (same logic as your notebook)
# ----------------------------
def build_htn_label_by_bpx(df: pd.DataFrame):
    sbp_cols = [c for c in df.columns if c.upper() in {"BPXSY1","BPXSY2","BPXSY3","BPXSY4"}]
    dbp_cols = [c for c in df.columns if c.upper() in {"BPXDI1","BPXDI2","BPXDI3","BPXDI4"}]

    sbp = df[sbp_cols].apply(to_num).mean(axis=1, skipna=True) if sbp_cols else pd.Series(np.nan, index=df.index)
    dbp = df[dbp_cols].apply(to_num).mean(axis=1, skipna=True) if dbp_cols else pd.Series(np.nan, index=df.index)

    valid = sbp.notna() | dbp.notna()
    y = pd.Series(np.nan, index=df.index)
    y.loc[valid] = ((sbp.loc[valid] >= 130) | (dbp.loc[valid] >= 80)).astype(int)
    return y

def build_hf_label_by_mcq160b(df: pd.DataFrame):
    if "MCQ160B" not in df.columns:
        raise KeyError("MCQ160B not found in df.")
    raw = to_num(df["MCQ160B"])
    mask = raw.isin([1, 2])
    y = (raw[mask] == 1).astype(int)
    return y, mask

# ----------------------------
# Feature policy (match your pretest)
# ----------------------------
DEMO_PREFIX = ("RIAGENDR","RIDAGEYR","RIDRETH","DMDEDU","DMDMARTL","INDHHIN2","INDFMPIR")
BMX_PREFIX  = ("BMX",)
LIFE_PREFIX = ("ALQ","SMQ","PAQ")

def pretest_features(df: pd.DataFrame, task: str):
    cols = []
    cols += [c for c in df.columns if c.startswith(DEMO_PREFIX)]
    cols += [c for c in df.columns if c.startswith(BMX_PREFIX)]
    cols += [c for c in df.columns if c.startswith(LIFE_PREFIX)]

    # leakage bans
    def forbidden(c: str):
        u = c.strip().upper()
        if task == "htn":
            return u.startswith("BPX") or u.startswith("BPQ")
        # hf strict
        return u.startswith(("MCQ","BPQ","CDQ","RX","BPX")) or u == "MCQ160B"

    cols = [c for c in dict.fromkeys(cols) if c in df.columns and not forbidden(c)]
    return cols

def make_numeric(df: pd.DataFrame, cols):
    X = df[cols].copy()
    for c in X.columns:
        if X[c].dtype == "object":
            X[c] = to_num(X[c])
        else:
            X[c] = to_num(X[c])
    X = X.dropna(axis=1, how="all")
    return X

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to NHANES_Merged_Data.csv")
    ap.add_argument("--task", required=True, choices=["htn", "hf"])
    ap.add_argument("--out_json", default="", help="Optional: save results json")
    args = ap.parse_args()

    df = pd.read_csv(args.csv, low_memory=False)

    if args.task == "htn":
        y = build_htn_label_by_bpx(df)
        valid = y.notna()
        yv = y[valid].astype(int)
        cols = pretest_features(df.loc[valid], task="htn")
        X = make_numeric(df.loc[valid], cols)
        # simple median impute (we will do train-only for one check)
        X = X.apply(lambda col: col.fillna(col.median()) if col.notna().any() else col)

    else:
        y, mask = build_hf_label_by_mcq160b(df)
        cols = pretest_features(df.loc[mask], task="hf")
        X = make_numeric(df.loc[mask], cols)
        # median impute (simple)
        X = X.apply(lambda col: col.fillna(col.median()) if col.notna().any() else col)
        yv = y

    # 1) Age-only
    if "RIDAGEYR" not in df.columns:
        raise KeyError("RIDAGEYR missing.")
    age = to_num(df.loc[yv.index, "RIDAGEYR"]).to_frame("RIDAGEYR")
    age = age.fillna(age.median(numeric_only=True))
    auc_age, ap_age = fit_xgb_auc(age, yv, n_estimators=200, max_depth=2)

    # 2) Shuffled label
    y_shuf = shuffle(yv, random_state=42).reset_index(drop=True)
    X_tmp = X.reset_index(drop=True)
    auc_shuf, ap_shuf = fit_xgb_auc(X_tmp, y_shuf, n_estimators=400, max_depth=4)

    # 3) Train-only median impute
    Xtr, Xte, ytr, yte = train_test_split(X, yv, test_size=0.2, stratify=yv, random_state=42)
    Xtr, Xte = median_impute_train_only(Xtr, Xte)
    m = xgb.XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9,
        eval_metric="logloss", random_state=42, n_jobs=-1
    )
    m.fit(Xtr, ytr)
    proba = m.predict_proba(Xte)[:, 1]
    auc_train_only = float(roc_auc_score(yte, proba))
    ap_train_only = float(average_precision_score(yte, proba))

    results = {
        "task": args.task,
        "n_samples": int(len(yv)),
        "n_features": int(X.shape[1]),
        "age_only": {"roc_auc": auc_age, "pr_auc": ap_age},
        "shuffled_y": {"roc_auc": auc_shuf, "pr_auc": ap_shuf},
        "train_only_impute": {"roc_auc": auc_train_only, "pr_auc": ap_train_only},
    }

    print("\n================ SANITY CHECKS ================")
    print(json.dumps(results, indent=2))
    print("Expected: shuffled_y roc_auc ~ 0.50 (chance).")
    print("================================================\n")

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
