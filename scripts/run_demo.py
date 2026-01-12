# scripts/run_demo.py
import pandas as pd
from langchain_core.tracers.context import tracing_v2_enabled

from cdss.app_v21 import build_app


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="NHANES_Merged_Data.csv path")
    ap.add_argument("--artifacts-dir", default="artifacts")
    ap.add_argument("--project", default="ed-cdss-v21-2task-demo")
    ap.add_argument("--target-idx", type=int, default=14174)
    ap.add_argument("--mode", default="manual", choices=["manual", "high_risk", "high_risk_confirmed", "discordant"])
    ap.add_argument("--pipeline-task", default="hypertension_by_bp", choices=["hypertension_by_bp", "heart_failure_by_mcq"])
    ap.add_argument("--pool", type=int, default=5000)
    ap.add_argument("--min-risk", type=float, default=0.40)
    ap.add_argument("--prefer-stage2", action="store_true")
    ap.add_argument("--no-langsmith", action="store_true", help="Disable LangSmith tracing even if installed/configured.")
    args = ap.parse_args()

    df = pd.read_csv(args.csv, low_memory=False)

    # build app (loads models/explainers/artifacts inside)
    app = build_app(df=df, artifacts_dir=args.artifacts_dir)

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

    # --- LangSmith tracing (optional, fail-safe) ---
    if args.no_langsmith:
        res = app.invoke(payload, {"recursion_limit": 40})
    else:
        try:
            from observability.langsmith_setup import force_langsmith_us  # must exist in your repo
            ls_client = force_langsmith_us(project_name=args.project)

            with tracing_v2_enabled(project_name=args.project, client=ls_client):
                res = app.invoke(payload, {"recursion_limit": 40})

            print(f"\n✅ Trace Link: https://smith.langchain.com/o/me/projects?name={args.project}")
        except Exception as e:
            # If LangSmith isn't configured, demo should still run.
            print(f"[WARN] LangSmith tracing disabled due to error: {repr(e)}")
            res = app.invoke(payload, {"recursion_limit": 40})

    print(res.get("final_report", "(no report)"))


if __name__ == "__main__":
    main()
