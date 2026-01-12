# cdss/registry.py
from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import joblib


def _to_num(x):
    return pd.to_numeric(x, errors="coerce")


def _is_missing(x) -> bool:
    if x is None:
        return True
    if isinstance(x, float) and np.isnan(x):
        return True
    return False


@dataclass
class TaskConfig:
    task: str
    display: str
    model: Any
    explainer: Any | None
    feature_cols: List[str]
    core_cols: List[str]
    opt_cols: List[str]
    top_features_global: List[str]
    leakage_note: str
    label_col: str | None = None  # HF only


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_list(x) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(i) for i in x]
    return []


def build_medians(df_: pd.DataFrame, cols: List[str]) -> Dict[str, float]:
    cols = [c for c in cols if c in df_.columns]
    if not cols:
        return {}
    tmp = df_[cols].copy()
    for c in tmp.columns:
        if tmp[c].dtype == "object":
            tmp[c] = _to_num(tmp[c])
    return tmp.median(numeric_only=True).to_dict()


def cols_from_groups(df: pd.DataFrame, FEATURE_GROUPS: Dict[str, List[str]], groups: List[str]) -> List[str]:
    out: List[str] = []
    for g in groups:
        out += (FEATURE_GROUPS.get(g, []) or [])
    out = [c for c in dict.fromkeys(out) if c in df.columns]
    return out


def make_model_input(
    row_dict: Dict[str, Any],
    feature_cols: List[str],
    medians: Dict[str, float],
) -> Tuple[np.ndarray, int, int]:
    x = []
    miss = 0
    for c in feature_cols:
        v = _to_num(row_dict.get(c, np.nan))
        if _is_missing(v):
            miss += 1
            v = medians.get(c, 0.0)
        x.append(float(v))
    X = np.array(x, dtype=float).reshape(1, -1)
    return X, miss, len(feature_cols)


def _load_optional_joblib(path: str):
    return joblib.load(path) if os.path.exists(path) else None


def load_task_registry(
    df: pd.DataFrame,
    artifacts_dir: str,
    FEATURE_GROUPS: Dict[str, List[str]],
) -> Tuple[Dict[str, TaskConfig], Dict[str, Dict[str, float]]]:
    """
    Expects artifacts_dir contains (standard names from your training scripts):
      - htn_model.joblib
      - htn_explainer.joblib            (optional but recommended)
      - htn_artifact.json

      - hf_model.joblib
      - hf_explainer.joblib             (optional but recommended)
      - hf_artifact.json
    """

    # ----------------
    # HTN
    # ----------------
    htn_meta_path = os.path.join(artifacts_dir, "htn_artifact.json")
    htn_model_path = os.path.join(artifacts_dir, "htn_model.joblib")
    if not os.path.exists(htn_meta_path):
        raise FileNotFoundError(f"Missing: {htn_meta_path}")
    if not os.path.exists(htn_model_path):
        raise FileNotFoundError(f"Missing: {htn_model_path}")

    htn_meta = _load_json(htn_meta_path)
    htn_model = joblib.load(htn_model_path)

    htn_explainer = _load_optional_joblib(os.path.join(artifacts_dir, "htn_explainer.joblib"))

    TASK_HTN = str(htn_meta.get("task", "hypertension_by_bp"))
    htn_feature_cols = _safe_list(htn_meta.get("feature_cols"))
    htn_top_global = _safe_list(htn_meta.get("top_features_shap")) or _safe_list(htn_meta.get("top_features_perm"))

    # ----------------
    # HF
    # ----------------
    hf_meta_path = os.path.join(artifacts_dir, "hf_artifact.json")
    hf_model_path = os.path.join(artifacts_dir, "hf_model.joblib")
    if not os.path.exists(hf_meta_path):
        raise FileNotFoundError(f"Missing: {hf_meta_path}")
    if not os.path.exists(hf_model_path):
        raise FileNotFoundError(f"Missing: {hf_model_path}")

    hf_meta = _load_json(hf_meta_path)
    hf_model = joblib.load(hf_model_path)

    hf_explainer = _load_optional_joblib(os.path.join(artifacts_dir, "hf_explainer.joblib"))

    TASK_HF = str(hf_meta.get("task", "heart_failure_by_mcq"))
    hf_feature_cols = _safe_list(hf_meta.get("feature_cols"))
    hf_top_global = _safe_list(hf_meta.get("top_features_shap")) or _safe_list(hf_meta.get("top_features_perm"))
    hf_label_col = hf_meta.get("label_col", "MCQ160B")

    # ----------------
    # Core/Optional splits for missingness reporting
    # ----------------
    CORE_GROUPS = ["demographics", "anthropometry"]

    htn_core_cols = [c for c in cols_from_groups(df, FEATURE_GROUPS, CORE_GROUPS) if c in htn_feature_cols]
    htn_opt_cols = [c for c in cols_from_groups(df, FEATURE_GROUPS, ["questionnaire_lifestyle"]) if c in htn_feature_cols]

    hf_core_cols = [c for c in cols_from_groups(df, FEATURE_GROUPS, CORE_GROUPS) if c in hf_feature_cols]
    hf_opt_cols = [
        c
        for c in cols_from_groups(
            df,
            FEATURE_GROUPS,
            ["lab_lipids", "lab_glucose_diabetes", "lab_inflammation", "lab_cbc", "questionnaire_lifestyle"],
        )
        if c in hf_feature_cols
    ]

    registry: Dict[str, TaskConfig] = {
        TASK_HTN: TaskConfig(
            task=TASK_HTN,
            display="Hipertansiyon (BPX ile doğrulandı)",
            model=htn_model,
            explainer=htn_explainer,
            feature_cols=htn_feature_cols,
            core_cols=htn_core_cols,
            opt_cols=htn_opt_cols,
            top_features_global=htn_top_global[:20],
            leakage_note="PretestPolicy: BPX/BPQ hariç (sızdırmasız).",
            label_col=None,
        ),
        TASK_HF: TaskConfig(
            task=TASK_HF,
            display="Kalp Yetmezliği (MCQ geçmiş etiketi)",
            model=hf_model,
            explainer=hf_explainer,
            feature_cols=hf_feature_cols,
            core_cols=hf_core_cols,
            opt_cols=hf_opt_cols,
            top_features_global=hf_top_global[:20],
            # DIQ burada yok çünkü training tarafında DIQ'yu forbidden yapmamıştık
            leakage_note="PretestPolicy: MCQ/CDQ/BPX/BPQ/RX hariç (geçmiş/tanı sızıntısız).",
            label_col=str(hf_label_col) if hf_label_col else "MCQ160B",
        ),
    }

    medians_cache: Dict[str, Dict[str, float]] = {}
    for t, cfg in registry.items():
        medians_cache[t] = build_medians(df, cfg.feature_cols)

    return registry, medians_cache
