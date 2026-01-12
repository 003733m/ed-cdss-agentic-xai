# scripts/run_demo.py
import os, json
import pandas as pd
from langchain_core.tracers.context import tracing_v2_enabled

from observability.langsmith_setup import force_langsmith_us
from cdss.app_v21 import build_app

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="NHANES_Merged_Data.csv path")
    ap.add_argument("--artifacts-dir", default="artifacts")
    ap.add_argument("--project", default="ed-cdss-v21-2task-demo")
    ap.add_argument("--target-idx", type=int, default=14174)
    ap.add_argument("--mode", default="manual", choices=["manual","high_risk","high_risk_confirmed","discordant"])
    ap.add_argument("--pipeline-task", default="hypertension_by_bp", choices=["hypertension_by_bp","heart_failure_by_mcq"])
    ap.add_argument("--pool", type=int, default=5000)
    ap.add_argument("--min-risk", type=float, default=0.40)
    ap.add_argument("--prefer-stage2", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    # build app (loads models/explainers/artifacts inside)
    app = build_app(df=df, artifacts_dir=args.artifacts_dir)

    # langsmith
    ls_client = force_langsmith_us(project_name=args.project)

    payload = {
        "messages": [],
        "step": 0,
        "task": args.pipeline_task,
        "selection_mode": args.mode,
        "sample_n": args.pool,
        "min_risk": args.min_risk,
        "prefer_stage2": bool(args.prefer_stage2),
        "patient_index": args.target_idx,
        "per_task": {},
    }

    with tracing_v2_enabled(project_name=args.project, client=ls_client):
        res = app.invoke(payload, {"recursion_limit": 40})

    print(res.get("final_report", "(no report)"))
    print(f"\n✅ Trace Link: https://smith.langchain.com/o/me/projects?name={args.project}")

if __name__ == "__main__":
    main()
