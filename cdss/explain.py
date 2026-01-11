# cdss/explain.py
from __future__ import annotations

from typing import Dict, Any, List
import numpy as np
import pandas as pd


def compute_local_shap(
    explainer: Any | None,
    feature_cols: List[str],
    x_1xD: np.ndarray,
    top_k: int = 10,
) -> Dict[str, Any]:
    if explainer is None:
        return {"ok": False, "reason": "no_shap_explainer", "rows": []}

    try:
        x_df = pd.DataFrame(x_1xD, columns=feature_cols)
        sv = explainer.shap_values(x_df)

        # binary classification cases:
        # - list of 2 arrays [neg, pos]
        # - array shape (N, D)
        if isinstance(sv, list) and len(sv) == 2:
            sv_pos = sv[1][0]
        else:
            arr = sv[0] if isinstance(sv, (list, tuple)) else sv
            arr_np = np.array(arr)
            if arr_np.ndim == 2:
                sv_pos = arr_np[0]
            else:
                sv_pos = arr_np

        vals = x_df.iloc[0].values
        df_local = pd.DataFrame({"feature": feature_cols, "value": vals, "shap": sv_pos})
        df_local["abs"] = df_local["shap"].abs()
        df_local = df_local.sort_values("abs", ascending=False).head(top_k).copy()
        df_local["direction"] = df_local["shap"].apply(lambda z: "↑ risk" if z > 0 else ("↓ risk" if z < 0 else "0"))
        return {"ok": True, "rows": df_local[["feature", "value", "shap", "direction"]].to_dict("records")}
    except Exception as e:
        return {"ok": False, "reason": repr(e), "rows": []}
