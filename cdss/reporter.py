# cdss/reporter.py
from __future__ import annotations

from typing import Dict, Any, List
import numpy as np
import pandas as pd

from .registry import _to_num, _is_missing


def _fmt(v, unit: str = "") -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "NA"
    try:
        fv = float(v)
        if abs(fv) >= 100:
            s = f"{fv:.0f}"
        elif abs(fv) >= 10:
            s = f"{fv:.1f}"
        else:
            s = f"{fv:.2f}"
        return f"{s}{(' ' + unit).strip() if unit else ''}".strip()
    except Exception:
        return str(v)


def build_snapshot(row: Dict[str, Any]) -> Dict[str, Any]:
    pick = [
        ("RIDAGEYR", "Yaş_yıl", ""),
        ("RIAGENDR", "Cinsiyet_kodu", ""),
        ("BMXHT", "Boy_cm", "cm"),
        ("BMXWT", "Kilo_kg", "kg"),
        ("BMXBMI", "VKİ", ""),
        ("INDFMPIR", "GelirPIR", ""),
        ("SMQ020", "Sigara_100adet", ""),
        ("SMQ050Q", "Şu_anda_sigara_içiyor", ""),
    ]
    snap = {}
    for code, nice, unit in pick:
        if code in row:
            v = _to_num(row.get(code, np.nan))
            if not _is_missing(v):
                snap[nice] = _fmt(v, unit)
    return snap


def build_labs_table_md(row: Dict[str, Any]) -> str:
    LAB_CANDIDATES = [
        ("LBXGLU", "Glucose", "mg/dL"),
        ("LBXGH", "HbA1c", "%"),
        ("LBXTC", "TotalChol", "mg/dL"),
        ("LBXTR", "Triglyceride", "mg/dL"),
        ("LBDHDD", "HDL", "mg/dL"),
        ("LBXSCR", "Creatinine", "mg/dL"),
        ("LBXBUN", "BUN", "mg/dL"),
        ("LBXHSCRP", "hsCRP", "mg/L"),
        ("LBXHGB", "Hemoglobin", "g/dL"),
        ("LBXHCT", "Hematocrit", "%"),
        ("LBXPLTSI", "Trombosit", "hücre/uL"),
        ("LBXWBCSI", "Akyuvar", "hücre/uL"),
    ]
    lab_rows = []
    for code, name, unit in LAB_CANDIDATES:
        if code in row:
            v = _to_num(row.get(code, np.nan))
            if not _is_missing(v):
                lab_rows.append({"Test": name, "Kod": code, "Değer": _fmt(v, unit)})
    return pd.DataFrame(lab_rows).to_markdown(index=False) if lab_rows else "(Bu hasta için lab değeri yok)"


def _md_table_single_row(d: Dict[str, Any]) -> str:
    if not d:
        return "(yok)"
    return pd.DataFrame([d]).to_markdown(index=False)


def render_local_shap_md(shap_local: Dict[str, Any]) -> str:
    if shap_local.get("ok") and shap_local.get("rows"):
        df_ls = pd.DataFrame(shap_local["rows"])
        if "value" in df_ls.columns:
            df_ls["value"] = df_ls["value"].map(lambda x: _fmt(x, ""))
        if "shap" in df_ls.columns:
            df_ls["shap"] = df_ls["shap"].map(lambda x: _fmt(x, ""))
        return df_ls.to_markdown(index=False)
    return f"(Local SHAP hesaplanamadı: {shap_local.get('reason', 'NA')})"


def confirm_text(conf: Dict[str, Any]) -> str:
    if not conf.get("available"):
        return f"Confirm unavailable (type={conf.get('type')})"
    if conf.get("type") == "measurement":
        return (
            f"Doğrulama(BPX): HTN={conf.get('htn')} evre={conf.get('stage')} "
            f"SBP_ortalama={_fmt(conf.get('sbp_mean'), 'mmHg')} DBP_ortalama={_fmt(conf.get('dbp_mean'), 'mmHg')}"
        )
    return f"Doğrulama(Geçmiş): HF={conf.get('hf')} {conf.get('source_col')} aracılığıyla"


def calc_missing_split(row_dict: Dict[str, Any], core_cols: List[str], optional_cols: List[str]) -> Dict[str, Any]:
    core_total = len(core_cols)
    core_missing = sum(1 for c in core_cols if _is_missing(_to_num(row_dict.get(c, np.nan))))
    opt_total = len(optional_cols)
    opt_missing = sum(1 for c in optional_cols if _is_missing(_to_num(row_dict.get(c, np.nan))))
    overall_total = core_total + opt_total
    overall_missing = core_missing + opt_missing
    return {
        "core_total": core_total,
        "core_missing": core_missing,
        "core_missing_rate": float(core_missing / max(1, core_total)),
        "optional_total": opt_total,
        "optional_missing": opt_missing,
        "optional_missing_rate": float(opt_missing / max(1, opt_total)),
        "overall_total": overall_total,
        "overall_missing": overall_missing,
        "overall_missing_rate": float(overall_missing / max(1, overall_total)),
    }


def agreement_and_trust(task: str, band: str, conf: Dict[str, Any], miss_split: Dict[str, Any]) -> Dict[str, Any]:
    core_rate = float(miss_split.get("core_missing_rate", 1.0))
    opt_rate = float(miss_split.get("optional_missing_rate", 1.0))
    core_quality = max(0.0, 1.0 - core_rate)
    optional_quality = max(0.0, 1.0 - opt_rate)
    coverage = float(conf.get("coverage", 0.0) or 0.0)

    agree = "BELİRSİZ"
    agree_score = 0.8

    if not conf.get("available"):
        agree = "Veri Yok (confirm yapılamadı)"
        agree_score = 0.0
    else:
        if task == "hypertension_by_bp":
            htn = conf.get("htn")
            if band == "HIGH" and htn is True:
                agree = "UYUMLU (Model HIGH → KB yüksek, HTN doğrulandı)"; agree_score = 1.0
            elif band == "LOW" and htn is False:
                agree = "UYUMLU (Model LOW → KB normal)"; agree_score = 1.0
            elif band == "HIGH" and htn is False:
                agree = "UYUMSUZ (Model HIGH ama KB normal)"; agree_score = 0.6
            elif band == "LOW" and htn is True:
                agree = "UYUMSUZ (Model LOW ama KB yüksek)"; agree_score = 0.6
            else:
                agree = "BELİRSİZ (MODERATE risk bandı)"; agree_score = 0.8
        else:
            hf = conf.get("hf")
            if band == "HIGH" and hf is True:
                agree = "UYUMLU (Model HIGH → HF history positive)"; agree_score = 1.0
            elif band == "LOW" and hf is False:
                agree = "UYUMLU (Model LOW → HF history negative)"; agree_score = 1.0
            elif band == "HIGH" and hf is False:
                agree = "UYUMSUZ (Model HIGH ama HF history negative)"; agree_score = 0.6
            elif band == "LOW" and hf is True:
                agree = "UYUMSUZ (Model LOW ama HF history positive)"; agree_score = 0.6
            else:
                agree = "BELİRSİZ (MODERATE risk bandı)"; agree_score = 0.8

    trust = (0.50 * core_quality + 0.20 * optional_quality + 0.20 * coverage + 0.10 * agree_score)
    trust_index = float(np.clip(trust, 0.0, 1.0))
    return {"agreement": agree, "trust_index": trust_index}


def render_report_two_tasks(
    patient_idx: int,
    selection_why: str,
    row: Dict[str, Any],
    per_task: Dict[str, Any],
    task_order: List[str],
    task_display: Dict[str, str],
    task_global_top: Dict[str, List[str]],
    leakage_notes: Dict[str, str],
    confirm_tables_md: Dict[str, str],
) -> str:
    snap = build_snapshot(row)
    snapshot_md = _md_table_single_row(snap)
    labs_md = build_labs_table_md(row)

    # Summary table
    summary_rows = []
    for t in task_order:
        pt = per_task[t]
        band_tr = {"LOW": "DÜŞÜK", "MODERATE": "ORTA", "HIGH": "YÜKSEK"}.get(pt["pretest_band"], pt["pretest_band"])
        summary_rows.append({
            "Görev": task_display[t],
            "PretestRisk": float(pt["pretest_risk"]),
            "Bant": band_tr,
            "Doğrulama": confirm_text(pt["confirm"]),
            "Uyum": pt["agreement"],
            "GüvenEndeksi": float(pt["trust_index"]),
        })
    sdf = pd.DataFrame(summary_rows)
    sdf["PretestRisk"] = sdf["PretestRisk"].map(lambda x: f"{x:.3f}")
    sdf["GüvenEndeksi"] = sdf["GüvenEndeksi"].map(lambda x: f"{x:.2f}")
    summary_md = sdf.to_markdown(index=False)

    blocks = []
    for t in task_order:
        pt = per_task[t]
        conf = pt["confirm"]
        miss = pt["missing_split"] or {}

        local_shap_note = ""
        if pt.get("shap_local", {}).get("ok") and pt.get("shap_local", {}).get("rows"):
            local_shap_note = "> Not: Yerel SHAP, **bu hasta özelinde** hangi değişkenlerin riski ↑/↓ ittiğini gösterir."

        core_m, core_t = int(miss.get("core_missing", 0)), int(miss.get("core_total", 0))
        opt_m, opt_t = int(miss.get("optional_missing", 0)), int(miss.get("optional_total", 0))
        overall_m, overall_t = int(miss.get("overall_missing", 0)), int(miss.get("overall_total", 1))
        model_m, model_t = int(miss.get("model_missing_n", 0)), int(miss.get("model_missing_total", 1))

        cdss_block = f"""
### CDSS Metrikleri ({task_display[t]})
- Görev: {t}
- {leakage_notes[t]}
- ÖzellikSayısı(model): {model_t}
- Eksiklik(temel): {core_m}/{core_t} (oran={(core_m/max(1,core_t)):.2f})
- Eksiklik(isteğe bağlı): {opt_m}/{opt_t} (oran={(opt_m/max(1,opt_t)):.2f})
- Eksiklik(genel): {overall_m}/{overall_t} (oran={(overall_m/max(1,overall_t)):.2f})
- Eksiklik(model girdileri): {model_m}/{model_t} (oran={(model_m/max(1,model_t)):.2f})
- DoğrulamaKapsamı: {conf.get("coverage", 0.0)}
- DoğrulamaSonucu: {confirm_text(conf)}
- Uyum: {pt.get("agreement","")}
- GüvenEndeksi(0-1): {float(pt.get("trust_index",0.0)):.2f}
""".strip()

        interp_lines = [
            f"- Model bandı **{pt['pretest_band']}** (risk={pt['pretest_risk']:.3f}).",
            (f"- Doğrulama katmanı mevcut: **{confirm_text(conf)}**." if conf.get("available")
             else "- Doğrulama katmanı mevcut değil → sonuç **risk taraması** olarak yorumlanmalı."),
            f"- Uyum değerlendirmesi: **{pt.get('agreement','')}**, GüvenEndeksi={float(pt.get('trust_index',0.0)):.2f}.",
        ]
        interp_md = "\n".join(interp_lines)

        if t == "hypertension_by_bp":
            action_md = """- Eğer HTN doğrulanıyorsa: tekrar ölçüm + klinik protokole göre değerlendirme
- Uyumsuzluk varsa: ev takibi/ABPM ile teyit (masked HTN / ölçüm varyasyonu)
- Yaşam tarzı: tuz azaltma, kilo kontrolü, egzersiz, sigara/alkol azaltma
"""
        else:
            action_md = """- HF geçmişi pozitif ise: semptom değerlendirme + kardiyoloji/eko yönlendirme (klinik protokole göre)
- Uyumsuzluk varsa: risk faktörleri / eksik veri nedeniyle model önyargısı olabilir → klinik doğrulama şart
- Komorbid tarama: böbrek fonksiyonu, anemi, diyabet, lipid (varsa lab ile destekle)
"""

        block = f"""
---

## {task_display[t]}

### A) Doğrulama Detayı
{confirm_tables_md[t]}

### B) Yerel Açıklama (Hastaya Özgü SHAP)
{local_shap_note}
{render_local_shap_md(pt.get("shap_local", {}) or {})}

### C) Küresel Özellik Önemi (Popülasyon Düzeyi)
> Not: Bu liste **hasta-bazlı açıklama değildir**; modelin genel olarak hangi değişkenlerden faydalandığını gösterir.
- {", ".join(task_global_top[t]) if task_global_top[t] else "(yok)"}

### D) Yorumlama (Demo)
{interp_md}

### E) Önerilen Klinik Eylem (Demo)
{action_md}

{cdss_block}
""".strip()
        blocks.append(block)

    final = f"""
# ED CDSS – Sunumluk Geniş Rapor (v21.1) – Tek Hasta, 2 Görev

> **UYARI:** Bu çıktı bir tanı değildir; klinik karar desteği sağlar. Nihai karar hekimin sorumluluğundadır.

## 1) Hasta Kimliği ve Seçim
- **Hasta (ILOC):** {patient_idx}
- **Seçim:** {selection_why}

## 2) Kısa Özet Tablosu (HTN + HF)
{summary_md}

## 3) Hasta Anlık Görüntüsü (Ön Test Görünür Özeti)
{snapshot_md}

## 4) Ek Laboratuvar Bulguları (Varsa)
{labs_md}

{"\n\n".join(blocks)}

> Laboratuvarlar **model girdisi değildir**; sadece klinik bağlam içindir.

**Bu bir tanı değildir.**
""".strip()
    return final
