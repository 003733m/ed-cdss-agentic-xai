#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
scripts/export_figures.py
Exports SHAP summary / dependence plots into docs/figures for paper.

Expected artifacts:
- artifacts/htn_model.joblib
- artifacts/htn_explainer.joblib
- artifacts/htn_feature_cols.json
- artifacts/hf_model.joblib
- artifacts/hf_explainer.joblib
- artifacts/hf_feature_cols.json

Usage:
  python scripts/export_figures.py --csv /path/NHANES_Merged_Data.csv --out docs/figures
"""

import argparse
import os
import json
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import shap
import joblib

def to_num(s): return pd.to_numeric(s, errors="coerce")

def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def make_numeric(df: pd.DataFrame, cols):
    X = df[cols].copy()
    for c in X.columns:
        X[c] = to_num(X[c])
    X = X.dropna(axis=1, how="all")
    X = X.apply(lambda col: col.fillna(col.median()) if col.notna().any() else col)
    return X

def export_task(task_name, df, out_dir, artifacts_dir, dependence_features):
    model_p = os.path.join(artifacts_dir, f"{task_name}_model.joblib")
    expl_p  = os.path.join(artifacts_dir, f"{task_name}_explainer.joblib")
    cols_p  = os.path.join(artifacts_dir, f"{task_name}_feature_cols.json")

    if not (os.path.exists(model_p) and os.path.exists(expl_p) and os.path.exists(cols_p)):
        raise FileNotFoundError(f"Missing artifacts for {task_name}. Expected: {model_p}, {expl_p}, {cols_p}")

    model = joblib.load(model_p)
    explainer = joblib.load(expl_p)
    cols = load_json(cols_p)

    X = make_numeric(df, cols)
    # sample for speed
    Xs = X.sample(n=min(500, len(X)), random_state=42)

    # shap values
    sv = explainer.shap_values(Xs)
    if isinstance(sv, list) and len(sv) == 2:
        sv_pos = sv[1]
    else:
        sv_pos = sv

    # 1) SHAP summary plot
    plt.figure()
    shap.summary_plot(sv_pos, Xs, max_display=15, show=False)
    plt.title(f"SHAP Summary ({task_name})")
    p1 = os.path.join(out_dir, f"{task_name}_shap_summary.png")
    plt.tight_layout()
    plt.savefig(p1, dpi=200, bbox_inches="tight")
    plt.close()

    # 2) Dependence plots
    for f in dependence_features:
        if f not in Xs.columns:
            continue
        plt.figure()
        shap.dependence_plot(f, sv_pos, Xs, show=False)
        plt.title(f"SHAP Dependence: {f} ({task_name})")
        p = os.path.join(out_dir, f"{task_name}_shap_dependence_{f}.png")
        plt.tight_layout()
        plt.savefig(p, dpi=200, bbox_inches="tight")
        plt.close()

    return {"task": task_name, "n": int(len(X)), "saved_summary": p1}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="docs/figures")
    ap.add_argument("--artifacts", default="artifacts")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    df = pd.read_csv(args.csv, low_memory=False)

    r1 = export_task("htn", df, args.out, args.artifacts, dependence_features=["RIDAGEYR","BMXWT","BMXBMI"])
    r2 = export_task("hf",  df, args.out, args.artifacts, dependence_features=["RIDAGEYR","BMXWT","BMXWAIST"])

    print("✅ Export done:")
    print(r1)
    print(r2)
    print(f"Figures saved in: {args.out}")

if __name__ == "__main__":
    main()
