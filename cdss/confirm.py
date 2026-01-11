# cdss/confirm.py
from __future__ import annotations

from typing import Dict, Any
import numpy as np
import pandas as pd

from .registry import _to_num, _is_missing


def compute_htn_confirm(row: Dict[str, Any]) -> Dict[str, Any]:
    sbp_cols = [f"BPXSY{i}" for i in [1, 2, 3, 4] if f"BPXSY{i}" in row]
    dbp_cols = [f"BPXDI{i}" for i in [1, 2, 3, 4] if f"BPXDI{i}" in row]

    def mean_cols(cols):
        xs = []
        for c in cols:
            v = _to_num(row.get(c, np.nan))
            if not _is_missing(v):
                xs.append(float(v))
        return float(np.mean(xs)) if xs else None

    sbp = mean_cols(sbp_cols)
    dbp = mean_cols(dbp_cols)

    if sbp is None and dbp is None:
        return {
            "available": False,
            "type": "measurement",
            "htn": None,
            "stage": None,
            "sbp_mean": None,
            "dbp_mean": None,
            "coverage": 0.0,
        }

    htn = ((sbp is not None and sbp >= 130) or (dbp is not None and dbp >= 80))
    stage = 0
    if (sbp is not None and sbp >= 140) or (dbp is not None and dbp >= 90):
        stage = 2
    elif ((sbp is not None and 130 <= sbp < 140) or (dbp is not None and 80 <= dbp < 90)):
        stage = 1

    have_sbp = sbp is not None
    have_dbp = dbp is not None
    coverage = 1.0 if (have_sbp and have_dbp) else (0.6 if (have_sbp or have_dbp) else 0.0)

    return {
        "available": True,
        "type": "measurement",
        "htn": bool(htn),
        "stage": int(stage),
        "sbp_mean": sbp,
        "dbp_mean": dbp,
        "coverage": float(coverage),
    }


def compute_hf_confirm(row: Dict[str, Any], label_col: str) -> Dict[str, Any]:
    v = _to_num(row.get(label_col, np.nan))
    if _is_missing(v) or v not in [1, 2]:
        return {"available": False, "type": "history", "hf": None, "source_col": label_col, "coverage": 0.0}
    return {"available": True, "type": "history", "hf": bool(v == 1), "source_col": label_col, "coverage": 1.0}


def build_bp_table(row: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for i in [1, 2, 3, 4]:
        sy = _to_num(row.get(f"BPXSY{i}", np.nan))
        di = _to_num(row.get(f"BPXDI{i}", np.nan))
        if _is_missing(sy) and _is_missing(di):
            continue
        rows.append({"Reading": i, "SBP_mmHg": sy, "DBP_mmHg": di})
    return pd.DataFrame(rows, columns=["Reading", "SBP_mmHg", "DBP_mmHg"])
