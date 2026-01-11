# cdss/app_v21.py
from __future__ import annotations

from typing import TypedDict, List, Dict, Any
import numpy as np
import pandas as pd

from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage

from .features import build_feature_groups
from .registry import load_task_registry, make_model_input
from .confirm import compute_htn_confirm, compute_hf_confirm, build_bp_table
from .explain import compute_local_shap
from .reporter import (
    calc_missing_split,
    agreement_and_trust,
    render_report_two_tasks,
    _fmt,
)

# ----------------------------
# STATE
# ----------------------------
class CDSSState(TypedDict, total=False):
    messages: List[Any]
    next_agent: str
    step: int

    task: str
    selection_mode: str
    sample_n: int
    min_risk: float
    prefer_stage2: bool
    patient_index: int
    selection_why: str

    hidden_data: Dict[str, Any]
    per_task: Dict[str, Any]
    final_report: str


def build_app(df: pd.DataFrame, artifacts_dir: str):
    # Feature groups for core/opt splits (same as notebook)
    FEATURE_GROUPS, _ = build_feature_groups(df)

    # Load models + explainers + artifact metadata; build median cache
    TASK_REGISTRY, MEDIAN_CACHE = load_task_registry(df, artifacts_dir, FEATURE_GROUPS)

    # Identify tasks (expect exactly these two keys)
    # Keep stable ordering for report
    TASK_HTN = "hypertension_by_bp" if "hypertension_by_bp" in TASK_REGISTRY else list(TASK_REGISTRY.keys())[0]
    TASK_HF = "heart_failure_by_mcq" if "heart_failure_by_mcq" in TASK_REGISTRY else list(TASK_REGISTRY.keys())[-1]

    task_order = [TASK_HTN, TASK_HF]

    def supervisor_node(state: CDSSState):
        step = int(state.get("step", 0))
        steps = {0: "patient_agent", 1: "predictor_agent", 2: "validator_agent", 3: "reporter_agent"}
        return {"next_agent": steps.get(step, "FINISH")}

    # ----------------------------
    # Scan selection (optional)
    # ----------------------------
    def scan_pick_patient(task: str, mode: str, pool: int, min_risk: float, prefer_stage2: bool, seed: int = 0) -> Dict[str, Any]:
        rng = np.random.default_rng(seed)
        n = len(df)
        pool = int(min(max(1, pool), n))
        idxs = rng.choice(n, size=pool, replace=False)

        cfg = TASK_REGISTRY[task]
        model = cfg.model

        best = None
        for idx in idxs:
            row = df.iloc[int(idx)].to_dict()
            X1, _, _ = make_model_input(row, cfg.feature_cols, MEDIAN_CACHE[task])
            try:
                risk = float(model.predict_proba(X1)[0, 1])
            except Exception:
                continue

            if risk < min_risk:
                continue

            # confirm
            if task == TASK_HTN:
                conf = compute_htn_confirm(row)
            else:
                conf = compute_hf_confirm(row, cfg.label_col or "MCQ160B")

            if mode == "high_risk":
                score = risk

            elif mode == "high_risk_confirmed":
                if task == TASK_HTN:
                    if not conf.get("available") or conf.get("htn") is not True:
                        continue
                    stage = conf.get("stage", None)
                    score = risk - (0.05 if (prefer_stage2 and stage != 2) else 0.0)
                else:
                    if not conf.get("available") or conf.get("hf") is not True:
                        continue
                    score = risk

            elif mode == "discordant":
                if not conf.get("available"):
                    continue
                if task == TASK_HTN:
                    if conf.get("htn") is True:
                        continue
                else:
                    if conf.get("hf") is True:
                        continue
                score = risk

            else:
                break

            if (best is None) or (score > best["score"]):
                best = {"idx": int(idx), "risk": risk, "conf": conf, "score": score}

        if best is None:
            return {"ok": False, "reason": f"no patient found in pool={pool} for mode={mode}", "idx": None}
        return {"ok": True, "idx": best["idx"], "risk": best["risk"], "reason": f"scan:{mode} (pool={pool})"}

    # ----------------------------
    # Agents
    # ----------------------------
    def patient_agent(state: CDSSState):
        task = state.get("task", TASK_HTN)
        if task not in TASK_REGISTRY:
            task = TASK_HTN

        mode = state.get("selection_mode", "manual")
        pool = int(state.get("sample_n", 5000))
        min_risk = float(state.get("min_risk", 0.40))
        prefer_stage2 = bool(state.get("prefer_stage2", True))

        idx = int(state.get("patient_index", 0))
        why = f"Manuel (TARGET_IDX={idx})"

        if mode != "manual":
            pick = scan_pick_patient(task, mode, pool, min_risk, prefer_stage2, seed=0)
            if pick.get("ok"):
                idx = int(pick["idx"])
                why = pick["reason"]
            else:
                why = f"Manual fallback (scan failed: {pick.get('reason')})"

        idx = max(0, min(idx, len(df) - 1))
        row = df.iloc[idx].to_dict()

        return {
            "step": 1,
            "task": task,
            "patient_index": idx,
            "selection_why": why,
            "hidden_data": row,
            "per_task": dict(state.get("per_task", {}) or {}),
            "messages": [AIMessage(content=f"Patient loaded idx={idx} | pipeline_task={task} | mode={mode}")],
        }

    def _ensure_task(per: Dict[str, Any], row: Dict[str, Any], task_name: str):
        need_keys = ["pretest_risk", "pretest_band", "missing_split", "confirm", "agreement", "trust_index", "shap_local"]
        if task_name in per and all(k in per[task_name] for k in need_keys):
            return

        cfg = TASK_REGISTRY[task_name]
        X1, miss_n, miss_t = make_model_input(row, cfg.feature_cols, MEDIAN_CACHE[task_name])

        try:
            risk = float(cfg.model.predict_proba(X1)[0, 1])
        except Exception:
            risk = 0.0

        band = "LOW" if risk < 0.15 else ("MODERATE" if risk < 0.40 else "HIGH")

        shap_local = compute_local_shap(cfg.explainer, cfg.feature_cols, X1, top_k=10)

        miss_split = calc_missing_split(row, cfg.core_cols, cfg.opt_cols)
        miss_split["model_missing_n"] = int(miss_n)
        miss_split["model_missing_total"] = int(miss_t)
        miss_split["model_missing_rate"] = float(miss_n / max(1, miss_t))

        if task_name == TASK_HTN:
            conf = compute_htn_confirm(row)
        else:
            conf = compute_hf_confirm(row, cfg.label_col or "MCQ160B")

        at = agreement_and_trust(task_name, band, conf, miss_split)

        per[task_name] = {
            "pretest_risk": float(risk),
            "pretest_band": band,
            "shap_local": shap_local,
            "missing_split": miss_split,
            "confirm": conf,
            "agreement": at["agreement"],
            "trust_index": at["trust_index"],
        }

    def predictor_agent(state: CDSSState):
        # We compute only for selected task; reporter will ensure both
        task = state["task"]
        row = state["hidden_data"]
        per = dict(state.get("per_task", {}) or {})

        _ensure_task(per, row, task)
        return {"step": 2, "per_task": per}

    def validator_agent(state: CDSSState):
        # already included in _ensure_task; keep for graph step consistency
        return {"step": 3}

    def _render_confirm_table_md(task_name: str, row: Dict[str, Any]) -> str:
        if task_name == TASK_HTN:
            bp_df = build_bp_table(row)
            if len(bp_df) == 0:
                return "(BPX readings not available)"
            bp_show = bp_df.copy()
            bp_show["SBP_mmHg"] = bp_show["SBP_mmHg"].map(lambda x: _fmt(x, "mmHg"))
            bp_show["DBP_mmHg"] = bp_show["DBP_mmHg"].map(lambda x: _fmt(x, "mmHg"))
            bp_show = bp_show.rename(columns={"Reading": "Ölçüm"})
            return bp_show.to_markdown(index=False)

        cfg = TASK_REGISTRY[task_name]
        col = cfg.label_col or "MCQ160B"
        raw = row.get(col, np.nan)
        meaning = "1=Evet, 2=Hayır (diğerleri/NA => bilinmiyor)"
        return pd.DataFrame([{"KY_geçmiş_sütunu": col, "ham_değer": raw, "anlam": meaning}]).to_markdown(index=False)

    def reporter_agent(state: CDSSState):
        idx = int(state.get("patient_index", 0))
        row = state.get("hidden_data", {}) or {}
        why = state.get("selection_why", f"Manuel (TARGET_IDX={idx})")

        per = dict(state.get("per_task", {}) or {})
        _ensure_task(per, row, TASK_HTN)
        _ensure_task(per, row, TASK_HF)

        task_display = {t: TASK_REGISTRY[t].display for t in task_order}
        task_global_top = {t: TASK_REGISTRY[t].top_features_global for t in task_order}
        leakage_notes = {t: TASK_REGISTRY[t].leakage_note for t in task_order}
        confirm_tables_md = {t: _render_confirm_table_md(t, row) for t in task_order}

        final = render_report_two_tasks(
            patient_idx=idx,
            selection_why=why,
            row=row,
            per_task=per,
            task_order=task_order,
            task_display=task_display,
            task_global_top=task_global_top,
            leakage_notes=leakage_notes,
            confirm_tables_md=confirm_tables_md,
        )
        return {"step": 4, "per_task": per, "final_report": final}

    # ----------------------------
    # Build graph
    # ----------------------------
    workflow = StateGraph(CDSSState)
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("patient_agent", patient_agent)
    workflow.add_node("predictor_agent", predictor_agent)
    workflow.add_node("validator_agent", validator_agent)
    workflow.add_node("reporter_agent", reporter_agent)

    workflow.set_entry_point("supervisor")
    nodes = ["patient_agent", "predictor_agent", "validator_agent", "reporter_agent"]
    workflow.add_conditional_edges("supervisor", lambda s: s["next_agent"], {n: n for n in nodes} | {"FINISH": END})
    for n in nodes:
        workflow.add_edge(n, "supervisor")

    return workflow.compile()
