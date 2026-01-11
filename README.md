# ED-NHANES CDSS (LangGraph) — HTN + HF (v21.1)

A reproducible, leakage-aware **Clinical Decision Support System (CDSS)** demo built on **NHANES** data.
The pipeline trains two “pretest” risk models and generates a **single-patient report** that includes:

- **HTN (Hypertension)**: label computed from BPX measurements (SBP/DBP)
- **HF (Heart Failure)**: label from NHANES medical condition questionnaire (MCQ160B)
- **Local explanations** via **SHAP** (patient-specific)
- **Global feature importance** via SHAP / permutation importance
- Optional **LangSmith tracing** for observability
- Optional **LLM polish** (Gemini) for report readability (no hallucination policy)

> ⚠️ **Disclaimer:** This project is a research/demo artifact. It does **not** provide medical diagnosis.  
> Final clinical decisions are the responsibility of licensed clinicians.

---

## Data Source

This project uses publicly available data from the **U.S. National Health and Nutrition Examination Survey (NHANES)**:

**NHANES 2017–2018 Continuous Cycle**  
CDC Website:  
https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?BeginYear=2017

The following NHANES files are used:
- DEMO (demographics, poverty ratio, age, sex)
- BPX / BPQ (blood pressure measurements and questionnaire)
- BMX (anthropometry)
- SMQ (smoking)
- MCQ (medical conditions, including heart failure)
- DIQ (diabetes)
- ALQ (alcohol)
- PAQ (physical activity)
- RXQ_RX (medication records)

These are merged using `DEMO` as the master table via `SEQN` left-joins.

---

## Repository Structure

```text
cdss/                       # Core CDSS package
  app_v21.py                # LangGraph workflow (patient → predict → validate → report)
  features.py               # Feature group builder (NHANES column grouping)
  registry.py               # Task registry (HTN/HF configs, models, columns)
  confirm.py                # Confirm layer (BPX confirmation, MCQ confirmation)
  explain.py                # Local SHAP utilities
  reporter.py               # Markdown report rendering

scripts/
  build_dataset.py          # Merge NHANES XPTs using DEMO as master (SEQN left-joins)
  train_htn_pretest.py      # Leakage-free HTN pretest training + artifact export
  train_hf_pretest.py       # Strict leakage-free HF pretest training + artifact export
  sanity_checks.py          # Age-only, shuffled-label, train-only-impute sanity checks
  export_figures.py         # Export SHAP plots to docs/figures (paper-ready)
  run_demo.py               # Run single-patient demo: produces 2-task report

docs/
  figures/                  # Paper-ready figures (SHAP summary / dependence plots)

notebooks/
  run_colab.ipynb           # End-to-end reproducibility notebook (Colab)

requirements.txt
README.md
.gitignore
