# cdss/features.py
import re
import pandas as pd

DIAG_LABELS = {
    "heart_failure": ["MCQ160B"],
    "coronary_heart_disease": ["MCQ160C"],
    "angina": ["MCQ160D"],
    "myocardial_infarction": ["MCQ160E"],
    "stroke": ["MCQ160F"],
    "family_early_heart_dz": ["MCQ300A"],
    "asthma_ever": ["MCQ010"],
    "asthma_current": ["MCQ035"],
    "asthma_attack_1y": ["MCQ040"],
    "asthma_ER_1y": ["MCQ050"],
    "emphysema": ["MCQ160G"],
    "chronic_bronchitis": ["MCQ160K"],
    "chronic_bronchitis_now": ["MCQ170K"],
    "copd": ["MCQ160O"],
    "diabetes_ever": ["DIQ010_x", "DIQ010_y"],
    "diabetes_doctor_confirm": ["DIQ160_x", "DIQ160_y"],
    "overweight_doctor": ["MCQ080"],
    "bp_high_doctor": ["BPQ020"],
    "bp_on_meds": ["BPQ040"],
    "chol_high_doctor": ["BPQ080"],
    "lipid_medication": ["BPQ100D"],
}

EXPLICIT_ALIAS_MAP = {
    "BPQ040": ["BPQ040", "BPQ040A", "BPQ040B"],
    "BPQ050": ["BPQ050", "BPQ050A", "BPQ050B"],
}

def _has_prefix(col: str, prefix: str) -> bool:
    return col.strip().upper().startswith(prefix.strip().upper())

def pick_first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def find_variants_in_df(df: pd.DataFrame, base_col: str) -> list[str]:
    base = base_col.strip().upper()

    if base_col in df.columns:
        return [base_col]

    for c in df.columns:
        if c.strip().upper() == base:
            return [c]

    pattern = re.compile(rf"^{re.escape(base)}([A-Z])?(_[XY])?$", re.IGNORECASE)
    hits = []
    for c in df.columns:
        cc = c.strip()
        if pattern.match(cc.upper()):
            hits.append(c)

    if not hits:
        hits = [c for c in df.columns if c.strip().upper().startswith(base)]
    return hits

def normalize_diag_labels(diag_labels: dict, df: pd.DataFrame, explicit_alias_map: dict | None = None) -> dict:
    explicit_alias_map = explicit_alias_map or {}
    norm = {}
    for task, cols in diag_labels.items():
        fixed_cols = []
        for col in cols:
            if col in df.columns:
                fixed_cols.append(col)
                continue

            if col in explicit_alias_map:
                picked = pick_first_existing(df, explicit_alias_map[col])
                if picked is not None:
                    fixed_cols.append(picked)
                    continue

            variants = find_variants_in_df(df, col)
            if variants:
                base = col.strip().upper()
                priority = [base, base + "A", base + "_X", base + "_Y"]
                chosen = None
                for p in priority:
                    for v in variants:
                        if v.strip().upper() == p:
                            chosen = v
                            break
                    if chosen:
                        break
                fixed_cols.append(chosen or variants[0])
                continue

        if fixed_cols:
            seen, kept = set(), []
            for c in fixed_cols:
                if c not in seen:
                    kept.append(c); seen.add(c)
            norm[task] = kept
    return norm

def build_feature_groups(df: pd.DataFrame) -> tuple[dict, list]:
    SURVEY_VARS = [c for c in ["WTMEC2YR", "WTINT2YR", "SDMVSTRA", "SDMVPSU"] if c in df.columns]

    demo_cols = [c for c in df.columns if c.startswith((
        "RIAGENDR", "RIDAGEYR", "RIDRETH", "DMDEDU", "DMDMARTL", "INDHHIN2", "INDFMPIR"
    ))]

    bp_cols = [c for c in df.columns if _has_prefix(c, "BPX")]
    bmx_cols = [c for c in df.columns if _has_prefix(c, "BMX")]

    lipid_cols = sorted(set(
        [c for c in df.columns if _has_prefix(c, "LBXTC")] +
        [c for c in df.columns if _has_prefix(c, "LBDTCSI")] +
        [c for c in df.columns if _has_prefix(c, "LBXTR")] +
        [c for c in df.columns if _has_prefix(c, "LBDTRSI")] +
        [c for c in df.columns if _has_prefix(c, "LBDHDD")] +
        [c for c in df.columns if _has_prefix(c, "LBDHDDS")] +
        [c for c in df.columns if _has_prefix(c, "LBDHDDSI")]
    ))

    glucose_prefixes = ["LBXGLU", "LBDGLUSI", "LBXGH", "LBXGHB", "LBXHG", "LBDGH", "LBXHBA", "LBXHA"]
    glucose_cols = sorted(set([c for c in df.columns if any(_has_prefix(c, p) for p in glucose_prefixes)]))

    inflam_cols = [c for c in df.columns if _has_prefix(c, "LBXHSCRP") or _has_prefix(c, "LBDHSCRP")]

    cbc_prefixes = ["LBXWBCSI","LBXLYPCT","LBXMNPCT","LBXNEPCT","LBXEOPCT","LBXBAPCT",
                    "LBXHGB","LBXHCT","LBXMCVSI","LBXPLTSI","LBXRDW"]
    cbc_cols = [c for c in df.columns if any(_has_prefix(c, p) for p in cbc_prefixes)]

    preg_cols = [c for c in df.columns if _has_prefix(c, "URXPREG") or _has_prefix(c, "UCPREG")]
    cdq_cols = [c for c in df.columns if _has_prefix(c, "CDQ")]
    bpq_cols = [c for c in df.columns if _has_prefix(c, "BPQ")]

    alq_cols = [c for c in df.columns if _has_prefix(c, "ALQ")]
    smq_cols = [c for c in df.columns if _has_prefix(c, "SMQ")]
    paq_cols = [c for c in df.columns if _has_prefix(c, "PAQ")]

    doctor_advice_cols = [c for c in df.columns if _has_prefix(c, "MCQ366") or _has_prefix(c, "MCQ371")]

    FEATURE_GROUPS = {
        "demographics": demo_cols,
        "vitals_bp": bp_cols,
        "anthropometry": bmx_cols,
        "lab_lipids": lipid_cols,
        "lab_glucose_diabetes": glucose_cols,
        "lab_inflammation": inflam_cols,
        "lab_cbc": cbc_cols,
        "pregnancy": preg_cols,
        "symptoms_chest_cardiac": cdq_cols,
        "questionnaire_bp_chol": bpq_cols,
        "questionnaire_lifestyle": sorted(set(alq_cols + smq_cols + paq_cols)),
        "doctor_advice_behavior": doctor_advice_cols,
    }
    return FEATURE_GROUPS, SURVEY_VARS
