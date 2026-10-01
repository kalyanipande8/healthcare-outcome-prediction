"""
Feature engineering for the healthcare outcome (30-day readmission) pipeline.

Reads the simulated EHR tables out of the local SQLite database (standing in
for a Databricks Delta Lake feature-engineering job) and builds a wide,
one-row-per-encounter feature table with 50+ distinct, real engineered
features spanning:
  - demographics
  - encounter-level features (length of stay, admission type)
  - comorbidity / diagnosis-derived counts and weighted scores
  - medication-class flags and polypharmacy counts
  - vital-sign temporal aggregations (mean/min/max/std/trend/first-last delta)
  - lab-value temporal aggregations and trend features per panel

All features are computed directly from the raw simulated tables -- none are
placeholders or duplicates of each other.
"""
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from src.generate_data import (
    COMORBIDITY_WEIGHTS, DB_PATH, LAB_PANELS, MED_CLASSES,
)

VITAL_COLS = ["heart_rate", "systolic_bp", "temperature", "spo2", "resp_rate"]


def _load_tables(db_path: Path):
    conn = sqlite3.connect(db_path)
    tables = {
        name: pd.read_sql(f"SELECT * FROM {name}", conn)
        for name in ["patients", "encounters", "vitals", "labs", "diagnoses", "medications", "outcomes"]
    }
    conn.close()
    return tables


def _trend_slope(values: np.ndarray) -> float:
    """Simple linear-regression slope of a value series against observation order."""
    n = len(values)
    if n < 2:
        return 0.0
    x = np.arange(n)
    x_mean, y_mean = x.mean(), values.mean()
    denom = ((x - x_mean) ** 2).sum()
    if denom == 0:
        return 0.0
    return float(((x - x_mean) * (values - y_mean)).sum() / denom)


def _demographic_features(patients: pd.DataFrame, encounters: pd.DataFrame) -> pd.DataFrame:
    df = encounters[["encounter_id", "patient_id"]].merge(patients, on="patient_id", how="left")
    out = pd.DataFrame({"encounter_id": df["encounter_id"]})
    out["age"] = df["age"]
    out["bmi"] = df["bmi"]
    out["is_male"] = (df["sex"] == "M").astype(int)
    out["is_medicare"] = (df["insurance"] == "Medicare").astype(int)
    out["is_medicaid"] = (df["insurance"] == "Medicaid").astype(int)
    out["is_uninsured"] = (df["insurance"] == "Uninsured").astype(int)
    out["age_over_65"] = (df["age"] >= 65).astype(int)
    out["age_over_80"] = (df["age"] >= 80).astype(int)
    out["bmi_obese"] = (df["bmi"] >= 30).astype(int)
    out["bmi_underweight"] = (df["bmi"] < 18.5).astype(int)
    return out


def _encounter_features(encounters: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({"encounter_id": encounters["encounter_id"]})
    out["length_of_stay"] = encounters["length_of_stay"]
    out["is_emergency_admission"] = (encounters["admission_type"] == "Emergency").astype(int)
    out["is_elective_admission"] = (encounters["admission_type"] == "Elective").astype(int)
    out["is_urgent_admission"] = (encounters["admission_type"] == "Urgent").astype(int)
    out["long_stay_flag"] = (encounters["length_of_stay"] >= 7).astype(int)
    out["short_stay_flag"] = (encounters["length_of_stay"] <= 2).astype(int)
    # prior-encounter count per patient (simple temporal history feature)
    enc_sorted = encounters.sort_values(["patient_id", "admit_date"]).copy()
    enc_sorted["prior_encounter_count"] = enc_sorted.groupby("patient_id").cumcount()
    out = out.merge(enc_sorted[["encounter_id", "prior_encounter_count"]], on="encounter_id", how="left")
    return out


def _comorbidity_features(diagnoses: pd.DataFrame, all_encounter_ids: pd.Series) -> pd.DataFrame:
    base = pd.DataFrame({"encounter_id": all_encounter_ids})
    dx_count = diagnoses.groupby("encounter_id").size().rename("diagnosis_count")
    weighted = diagnoses.assign(
        w=diagnoses["diagnosis_code"].map(COMORBIDITY_WEIGHTS).fillna(0.3)
    ).groupby("encounter_id")["w"].sum().rename("weighted_comorbidity_score")

    out = base.merge(dx_count, on="encounter_id", how="left").merge(weighted, on="encounter_id", how="left")
    out["diagnosis_count"] = out["diagnosis_count"].fillna(0)
    out["weighted_comorbidity_score"] = out["weighted_comorbidity_score"].fillna(0)

    # individual high-impact diagnosis flags (heart failure, CKD, COPD, etc.)
    flag_codes = {
        "has_heart_failure": "I50.9",
        "has_ckd": "N18.3",
        "has_copd": "J44.9",
        "has_diabetes": "E11.9",
        "has_cad": "I25.10",
        "has_depression": "F32.9",
        "has_afib": "I48.91",
        "has_aki": "N17.9",
        "has_stroke_history": "I63.9",
    }
    for feat_name, code in flag_codes.items():
        ids_with_code = set(diagnoses.loc[diagnoses["diagnosis_code"] == code, "encounter_id"])
        out[feat_name] = out["encounter_id"].isin(ids_with_code).astype(int)

    out["multimorbidity_flag"] = (out["diagnosis_count"] >= 3).astype(int)
    return out


def _medication_features(medications: pd.DataFrame, all_encounter_ids: pd.Series) -> pd.DataFrame:
    base = pd.DataFrame({"encounter_id": all_encounter_ids})
    med_count = medications.groupby("encounter_id").size().rename("medication_count")
    out = base.merge(med_count, on="encounter_id", how="left")
    out["medication_count"] = out["medication_count"].fillna(0)
    out["polypharmacy_flag"] = (out["medication_count"] >= 5).astype(int)

    for med_class in MED_CLASSES:
        ids_with_med = set(medications.loc[medications["med_class"] == med_class, "encounter_id"])
        out[f"on_{med_class}"] = out["encounter_id"].isin(ids_with_med).astype(int)
    return out


def _vital_features(vitals: pd.DataFrame, all_encounter_ids: pd.Series) -> pd.DataFrame:
    base = pd.DataFrame({"encounter_id": all_encounter_ids})
    agg_frames = [base]
    for col in VITAL_COLS:
        grp = vitals.groupby("encounter_id")[col]
        stats = grp.agg(["mean", "min", "max", "std"]).rename(
            columns={
                "mean": f"{col}_mean", "min": f"{col}_min",
                "max": f"{col}_max", "std": f"{col}_std",
            }
        )
        trend = grp.apply(lambda s: _trend_slope(s.to_numpy())).rename(f"{col}_trend")
        first_last = grp.apply(
            lambda s: (s.iloc[-1] - s.iloc[0]) if len(s) > 1 else 0.0
        ).rename(f"{col}_first_last_delta")
        agg_frames.append(stats.reset_index())
        agg_frames.append(trend.reset_index())
        agg_frames.append(first_last.reset_index())

    out = agg_frames[0]
    for f in agg_frames[1:]:
        out = out.merge(f, on="encounter_id", how="left")
    out = out.fillna(0)
    out["hypoxia_flag"] = (out["spo2_min"] < 90).astype(int)
    out["tachycardia_flag"] = (out["heart_rate_max"] > 100).astype(int)
    return out


def _lab_features(labs: pd.DataFrame, all_encounter_ids: pd.Series) -> pd.DataFrame:
    base = pd.DataFrame({"encounter_id": all_encounter_ids})
    agg_frames = [base]
    for panel in LAB_PANELS:
        sub = labs[labs["panel"] == panel]
        grp = sub.groupby("encounter_id")["value"]
        stats = grp.agg(["mean", "max"]).rename(
            columns={"mean": f"lab_{panel}_mean", "max": f"lab_{panel}_max"}
        )
        trend = grp.apply(lambda s: _trend_slope(s.to_numpy())).rename(f"lab_{panel}_trend")
        agg_frames.append(stats.reset_index())
        agg_frames.append(trend.reset_index())

    out = agg_frames[0]
    for f in agg_frames[1:]:
        out = out.merge(f, on="encounter_id", how="left")
    out = out.fillna(0)
    out["renal_risk_flag"] = (out["lab_creatinine_mean"] > 1.3).astype(int)
    out["anemia_flag"] = (out["lab_hemoglobin_mean"] < 12).astype(int)
    out["hyperglycemia_flag"] = (out["lab_glucose_mean"] > 140).astype(int)
    out["elevated_bnp_flag"] = (out["lab_bnp_mean"] > 100).astype(int)
    return out


def build_feature_table(db_path: Path = DB_PATH) -> pd.DataFrame:
    tables = _load_tables(db_path)
    encounters = tables["encounters"]
    all_ids = encounters["encounter_id"]

    demo = _demographic_features(tables["patients"], encounters)
    enc_feats = _encounter_features(encounters)
    comorb = _comorbidity_features(tables["diagnoses"], all_ids)
    meds = _medication_features(tables["medications"], all_ids)
    vitals = _vital_features(tables["vitals"], all_ids)
    labs = _lab_features(tables["labs"], all_ids)

    feats = demo
    for f in (enc_feats, comorb, meds, vitals, labs):
        feats = feats.merge(f, on="encounter_id", how="left")

    feats = feats.merge(tables["outcomes"], on="encounter_id", how="left")
    feats = feats.fillna(0)

    non_feature_cols = {"encounter_id", "readmitted_30d"}
    n_features = len([c for c in feats.columns if c not in non_feature_cols])
    print(f"Engineered feature table: {feats.shape[0]} encounters x {n_features} features")
    return feats


def get_feature_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in ("encounter_id", "readmitted_30d")]


if __name__ == "__main__":
    df = build_feature_table()
    out_path = Path(__file__).resolve().parent.parent / "data" / "features.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Saved feature table to {out_path}")
