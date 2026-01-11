# Paper Assets Checklist

## Reproducibility (Commands)
- Train HTN:
  python scripts/train_htn_pretest.py --csv <NHANES_Merged_Data.csv>

- Train HF:
  python scripts/train_hf_pretest.py --csv <NHANES_Merged_Data.csv>

- Sanity checks:
  python scripts/sanity_checks.py --csv <NHANES_Merged_Data.csv> --task htn
  python scripts/sanity_checks.py --csv <NHANES_Merged_Data.csv> --task hf

- Export figures:
  python scripts/export_figures.py --csv <NHANES_Merged_Data.csv> --out docs/figures

- Demo run:
  python scripts/run_demo.py --csv <NHANES_Merged_Data.csv> --target-idx 14174

## Figures to include
- docs/figures/htn_shap_summary.png
- docs/figures/htn_shap_dependence_RIDAGEYR.png
- docs/figures/hf_shap_summary.png
- docs/figures/hf_shap_dependence_RIDAGEYR.png

## Sanity results to cite in paper
- shuffled_y AUC ~ 0.50 (chance)
- age-only AUC baseline
- train-only imputation AUC
