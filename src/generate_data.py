"""
Simulate EHR-like patient data and store it in a local SQLite database.

This stands in for a Databricks Delta Lake / Lakehouse ingestion layer.
No real patient data is used -- everything here is synthetically generated
with numpy/pandas so the pipeline can be run and verified locally without
any managed infrastructure or PHI.

Tables produced (mirrors a simplified EHR schema):
  - patients:     one row per patient (demographics)
  - encounters:   hospital encounters/admissions
  - vitals:       time-stamped vital sign measurements per encounter
  - labs:         time-stamped lab results per encounter
  - diagnoses:    ICD-style diagnosis codes per encounter (for comorbidities)
  - medications:  medication orders per encounter (by drug class)
  - outcomes:     binary 30-day readmission label per encounter (target)
"""
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

RNG_SEED = 42
N_PATIENTS = 2500
DB_PATH = Path(__file__).resolve().parent.parent / "data" / "ehr.db"

DIAGNOSIS_CODES = [
    "I10", "E11.9", "N18.3", "J44.9", "I50.9", "E78.5", "I25.10", "F32.9",
    "N17.9", "J45.909", "K21.9", "M17.9", "G47.33", "E03.9", "D64.9",
    "I48.91", "E66.9", "M54.5", "I63.9", "N39.0",
]
COMORBIDITY_WEIGHTS = {  # higher weight => more severe comorbidity
    "I50.9": 3, "N18.3": 3, "I25.10": 2, "J44.9": 2, "E11.9": 1,
    "I63.9": 3, "F32.9": 1, "I48.91": 2, "N17.9": 3, "M17.9": 1,
}
MED_CLASSES = [
    "beta_blocker", "ace_inhibitor", "diuretic", "insulin", "statin",
    "anticoagulant", "opioid", "antibiotic", "antidepressant", "steroid",
    "bronchodilator", "antiplatelet",
]
LAB_PANELS = ["sodium", "potassium", "creatinine", "bun", "hemoglobin",
              "wbc", "glucose", "bnp", "troponin", "albumin"]


def _simulate_patients(rng):
    ages = rng.normal(64, 16, N_PATIENTS).clip(18, 98).round(1)
    sex = rng.choice(["M", "F"], size=N_PATIENTS)
    race = rng.choice(
        ["White", "Black", "Asian", "Hispanic", "Other"],
        size=N_PATIENTS, p=[0.55, 0.18, 0.09, 0.13, 0.05],
    )
    insurance = rng.choice(
        ["Medicare", "Medicaid", "Private", "Uninsured"],
        size=N_PATIENTS, p=[0.45, 0.2, 0.3, 0.05],
    )
    bmi = rng.normal(29, 6, N_PATIENTS).clip(15, 55).round(1)
    return pd.DataFrame({
        "patient_id": np.arange(1, N_PATIENTS + 1),
        "age": ages,
        "sex": sex,
        "race": race,
        "insurance": insurance,
        "bmi": bmi,
    })


def _simulate_encounters(rng, patients):
    rows = []
    enc_id = 1
    for pid in patients["patient_id"]:
        n_enc = rng.choice([1, 1, 1, 2, 2, 3], p=[0.35, 0.2, 0.15, 0.15, 0.1, 0.05])
        start = pd.Timestamp("2023-01-01") + pd.Timedelta(days=int(rng.integers(0, 500)))
        for _ in range(n_enc):
            los = int(rng.integers(1, 14))
            rows.append({
                "encounter_id": enc_id,
                "patient_id": pid,
                "admit_date": start,
                "discharge_date": start + pd.Timedelta(days=los),
                "length_of_stay": los,
                "admission_type": rng.choice(["Emergency", "Elective", "Urgent"], p=[0.55, 0.25, 0.2]),
            })
            start = start + pd.Timedelta(days=los + int(rng.integers(20, 200)))
            enc_id += 1
    return pd.DataFrame(rows)


def _simulate_vitals(rng, encounters):
    rows = []
    for _, enc in encounters.iterrows():
        n_obs = max(2, enc["length_of_stay"])
        base_hr = rng.normal(85, 12)
        base_sbp = rng.normal(128, 18)
        base_temp = rng.normal(37.0, 0.4)
        base_spo2 = rng.normal(96, 2)
        base_rr = rng.normal(18, 3)
        for t in range(n_obs):
            drift = t / max(n_obs - 1, 1)
            rows.append({
                "encounter_id": enc["encounter_id"],
                "obs_time": t,
                "heart_rate": base_hr + rng.normal(0, 5) + drift * rng.normal(0, 8),
                "systolic_bp": base_sbp + rng.normal(0, 8) + drift * rng.normal(0, 10),
                "temperature": base_temp + rng.normal(0, 0.2),
                "spo2": np.clip(base_spo2 + rng.normal(0, 1.5) - drift * rng.normal(0, 3), 70, 100),
                "resp_rate": base_rr + rng.normal(0, 2),
            })
    return pd.DataFrame(rows)


def _simulate_labs(rng, encounters):
    rows = []
    ref = {
        "sodium": (136, 145), "potassium": (3.5, 5.1), "creatinine": (0.6, 1.3),
        "bun": (7, 20), "hemoglobin": (12, 17), "wbc": (4, 11), "glucose": (70, 140),
        "bnp": (0, 100), "troponin": (0, 0.04), "albumin": (3.5, 5.0),
    }
    for _, enc in encounters.iterrows():
        n_obs = max(1, enc["length_of_stay"] // 2)
        for panel in LAB_PANELS:
            lo, hi = ref[panel]
            mid = (lo + hi) / 2
            spread = (hi - lo) / 2
            severity = rng.choice([0, 1, 2], p=[0.6, 0.3, 0.1])  # 0 normal .. 2 severe
            base = mid + severity * spread * rng.choice([-1, 1]) * rng.uniform(0.6, 1.8)
            trend = rng.normal(0, spread * 0.15)
            for t in range(n_obs):
                rows.append({
                    "encounter_id": enc["encounter_id"],
                    "panel": panel,
                    "obs_time": t,
                    "value": base + trend * t + rng.normal(0, spread * 0.1),
                })
    return pd.DataFrame(rows)


def _simulate_diagnoses(rng, encounters):
    rows = []
    for _, enc in encounters.iterrows():
        n_dx = rng.integers(1, 6)
        codes = rng.choice(DIAGNOSIS_CODES, size=n_dx, replace=False)
        for c in codes:
            rows.append({"encounter_id": enc["encounter_id"], "diagnosis_code": c})
    return pd.DataFrame(rows)


def _simulate_medications(rng, encounters):
    rows = []
    for _, enc in encounters.iterrows():
        n_meds = rng.integers(0, 6)
        if n_meds == 0:
            continue
        meds = rng.choice(MED_CLASSES, size=n_meds, replace=False)
        for m in meds:
            rows.append({"encounter_id": enc["encounter_id"], "med_class": m})
    return pd.DataFrame(rows)


def _simulate_outcomes(rng, encounters, diagnoses, medications, vitals, labs):
    """Derive a binary 30-day-readmission label from a latent risk score so the
    label is genuinely predictable from the engineered features, not random."""
    dx_counts = diagnoses.groupby("encounter_id").size()
    weighted_dx = diagnoses.assign(
        w=diagnoses["diagnosis_code"].map(COMORBIDITY_WEIGHTS).fillna(0.3)
    ).groupby("encounter_id")["w"].sum()
    med_counts = medications.groupby("encounter_id").size()
    hr_mean = vitals.groupby("encounter_id")["heart_rate"].mean()
    spo2_min = vitals.groupby("encounter_id")["spo2"].min()

    bnp = labs[labs["panel"] == "bnp"].groupby("encounter_id")["value"].mean()
    creat = labs[labs["panel"] == "creatinine"].groupby("encounter_id")["value"].mean()

    df = encounters.set_index("encounter_id")
    risk = (
        0.03 * df["age"].fillna(0) if "age" in df else 0
    )
    risk = pd.Series(0.0, index=df.index)
    risk += 0.05 * (df["length_of_stay"] - 5)
    risk += 1.6 * weighted_dx.reindex(df.index).fillna(0)
    risk += 0.35 * dx_counts.reindex(df.index).fillna(0)
    risk -= 0.05 * med_counts.reindex(df.index).fillna(0)  # some meds indicate mgmt, slight protective noise
    risk += 0.07 * (hr_mean.reindex(df.index).fillna(85) - 85)
    risk -= 0.18 * (spo2_min.reindex(df.index).fillna(96) - 96)
    risk += 0.05 * (bnp.reindex(df.index).fillna(50) - 50) / 10
    risk += 3.0 * (creat.reindex(df.index).fillna(1.0) - 1.0)
    risk += (df["admission_type"] == "Emergency").astype(float) * 0.9
    # modest noise -- keeps the task realistic (not perfectly separable) while
    # letting a well-tuned model recover most of the signal from the 100+
    # engineered features, similar to real clinical-risk-score performance.
    risk += rng.normal(0, 0.02, size=len(df))

    prob = 1 / (1 + np.exp(-(risk - risk.median())))
    label = (rng.uniform(0, 1, size=len(df)) < prob).astype(int)
    return pd.DataFrame({"encounter_id": df.index, "readmitted_30d": label.values})


def build_database(db_path: Path = DB_PATH, seed: int = RNG_SEED) -> Path:
    rng = np.random.default_rng(seed)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    patients = _simulate_patients(rng)
    encounters = _simulate_encounters(rng, patients)
    encounters = encounters.merge(patients[["patient_id", "age"]], on="patient_id", how="left")
    vitals = _simulate_vitals(rng, encounters)
    labs = _simulate_labs(rng, encounters)
    diagnoses = _simulate_diagnoses(rng, encounters)
    medications = _simulate_medications(rng, encounters)
    outcomes = _simulate_outcomes(rng, encounters, diagnoses, medications, vitals, labs)

    encounters = encounters.drop(columns=["age"])

    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(db_path)
    patients.to_sql("patients", conn, index=False)
    encounters.assign(
        admit_date=encounters["admit_date"].astype(str),
        discharge_date=encounters["discharge_date"].astype(str),
    ).to_sql("encounters", conn, index=False)
    vitals.to_sql("vitals", conn, index=False)
    labs.to_sql("labs", conn, index=False)
    diagnoses.to_sql("diagnoses", conn, index=False)
    medications.to_sql("medications", conn, index=False)
    outcomes.to_sql("outcomes", conn, index=False)
    conn.close()

    print(f"Simulated EHR database written to {db_path}")
    print(f"  patients:    {len(patients)}")
    print(f"  encounters:  {len(encounters)}")
    print(f"  vitals rows: {len(vitals)}")
    print(f"  labs rows:   {len(labs)}")
    print(f"  diagnoses:   {len(diagnoses)}")
    print(f"  medications: {len(medications)}")
    print(f"  outcomes:    {len(outcomes)}  (positive rate: {outcomes['readmitted_30d'].mean():.3f})")
    return db_path


if __name__ == "__main__":
    build_database()
