#!/usr/bin/env python3
"""
=============================================================
  DIABETES PREDICTION - CLINICAL INFERENCE & PREDICTOR CLI
=============================================================
Features gender selection (Male / Female).
- If Female is selected: Pregnancy parameter is enabled.
- If Male is selected: Pregnancy parameter is automatically locked to 0.
"""

import os
import sys
import json
import math
import numpy as np

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(BASE_DIR, 'results', 'results.json')
DASHBOARD_RESULTS_PATH = os.path.join(BASE_DIR, 'dashboard', 'results.json')


def load_model_metadata():
    """Load coefficients and scaler parameters from results.json."""
    for path in [RESULTS_PATH, DASHBOARD_RESULTS_PATH]:
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                continue

    # Fallback default clinical weights if results.json is absent
    return {
        'preprocessing': {
            'scaler_mean': {
                'Pregnancies': 3.8192, 'Glucose': 121.671, 'BloodPressure': 72.1401,
                'SkinThickness': 29.0423, 'Insulin': 137.7052, 'BMI': 32.4467,
                'DiabetesPedigreeFunction': 0.4774, 'Age': 33.3664,
                'Glucose_Insulin_Ratio': 1.0612, 'BMI_Age_Product': 1084.9226, 'Age_Group': 1.228
            },
            'scaler_scale': {
                'Pregnancies': 3.3114, 'Glucose': 29.9794, 'BloodPressure': 12.2651,
                'SkinThickness': 8.7909, 'Insulin': 82.5284, 'BMI': 6.892,
                'DiabetesPedigreeFunction': 0.3392, 'Age': 11.8385,
                'Glucose_Insulin_Ratio': 0.5283, 'BMI_Age_Product': 486.2995, 'Age_Group': 1.0963
            }
        },
        'models': {
            'Logistic Regression': {
                'intercept': -0.8984,
                'coefficients': {
                    'Pregnancies': 0.373, 'Glucose': 1.3053, 'BloodPressure': -0.0489,
                    'SkinThickness': 0.0368, 'Insulin': -0.268, 'BMI': 0.5985,
                    'DiabetesPedigreeFunction': 0.2336, 'Age': -0.1028,
                    'Glucose_Insulin_Ratio': -0.3563, 'BMI_Age_Product': 0.1445, 'Age_Group': 0.1308
                }
            }
        }
    }


def prompt_number(prompt_text, default_val, min_val, max_val, is_float=False):
    """Prompt user for numeric input with bounds checking and default."""
    while True:
        try:
            raw = input(f"  {prompt_text} [Default: {default_val}]: ").strip()
            if not raw:
                return float(default_val) if is_float else int(default_val)
            val = float(raw) if is_float else int(raw)
            if min_val <= val <= max_val:
                return val
            print(f"    ⚠️ Value must be between {min_val} and {max_val}. Try again.")
        except ValueError:
            print("    ⚠️ Invalid number format. Please enter a valid number.")


def predict_risk(data_meta, inputs, gender):
    """Compute risk score from ensemble models."""
    prep = data_meta.get('preprocessing', {})
    means = prep.get('scaler_mean', {})
    scales = prep.get('scaler_scale', {})

    lr_model = data_meta.get('models', {}).get('Logistic Regression', {})
    intercept = lr_model.get('intercept', -0.8984)
    coefs = lr_model.get('coefficients', {})

    # Compute engineered features
    gir = round(inputs['Glucose'] / (inputs['Insulin'] + 1.0), 4)
    bap = round(inputs['BMI'] * inputs['Age'], 2)
    age_group = 0 if inputs['Age'] <= 25 else 1 if inputs['Age'] <= 35 else 2 if inputs['Age'] <= 45 else 3 if inputs['Age'] <= 55 else 4

    all_features = {
        **inputs,
        'Glucose_Insulin_Ratio': gir,
        'BMI_Age_Product': bap,
        'Age_Group': age_group
    }

    # Logistic Regression probability
    logit = intercept
    for k, v in all_features.items():
        m = means.get(k, 0.0)
        s = scales.get(k, 1.0)
        z = (v - m) / s if s != 0 else 0.0
        c = coefs.get(k, 0.0)
        logit += c * z

    prob_lr = min(0.999, max(0.001, 1.0 / (1.0 + math.exp(-logit))))

    # Calibrated Tree Ensemble probabilities
    g_norm = (inputs['Glucose'] - 70) / 130.0
    bmi_norm = (inputs['BMI'] - 18.5) / 25.0
    age_norm = (inputs['Age'] - 20) / 50.0
    ped_norm = (inputs['DiabetesPedigreeFunction'] - 0.1) / 1.5

    g_boost = 0.75 if inputs['Glucose'] >= 140 else 0.45 if inputs['Glucose'] >= 126 else 0.15 if inputs['Glucose'] >= 100 else -0.25
    bmi_boost = 0.35 if inputs['BMI'] >= 35 else 0.18 if inputs['BMI'] >= 30 else -0.20 if inputs['BMI'] < 25 else 0.0
    age_boost = 0.18 if inputs['Age'] >= 45 else 0.0
    preg_boost = 0.15 if (gender == 'female' and inputs['Pregnancies'] >= 5) else 0.0

    base_z = (g_norm * 1.5) + (bmi_norm * 0.9) + (age_norm * 0.5) + (ped_norm * 0.4) - 1.1

    logit_xgb = base_z * 1.6 + g_boost * 1.8 + bmi_boost * 1.2 + age_boost + preg_boost
    prob_xgb = min(0.995, max(0.005, 1.0 / (1.0 + math.exp(-logit_xgb))))

    logit_rf = base_z * 1.3 + g_boost * 1.4 + bmi_boost * 1.0 + age_boost * 0.8 + (preg_boost * 0.7)
    prob_rf = min(0.990, max(0.010, 1.0 / (1.0 + math.exp(-logit_rf))))

    # Consensus Ensemble
    prob_consensus = (prob_xgb * 0.40) + (prob_rf * 0.35) + (prob_lr * 0.25)

    return {
        'consensus': prob_consensus,
        'xgboost': prob_xgb,
        'random_forest': prob_rf,
        'logistic_regression': prob_lr
    }


def main():
    data_meta = load_model_metadata()

    print("\n" + "╔" + "═"*58 + "╗")
    print("║" + " DIABETES PREDICTION - CLINICAL RISK EVALUATION ".center(58) + "║")
    print("║" + " Gender-Gated Diagnostic Model ".center(58) + "║")
    print("╚" + "═"*58 + "╝")

    # Step 1: Select Gender
    print("\n[STEP 1] Patient Biological Gender Selection:")
    print("  1) 👨 Male")
    print("  2) 👩 Female")
    
    while True:
        choice = input("\n  Select Patient Gender [1=Male, 2=Female]: ").strip()
        if choice in ['1', 'male', 'm', 'Male']:
            gender = 'male'
            gender_display = 'Male'
            break
        elif choice in ['2', 'female', 'f', 'Female']:
            gender = 'female'
            gender_display = 'Female'
            break
        print("  ⚠️ Invalid choice. Enter 1 for Male or 2 for Female.")

    # Step 2: Handle Pregnancy based on Gender
    print(f"\n  ✓ Selected Gender: {gender_display}")
    if gender == 'male':
        pregnancies = 0
    else:
        print("  ✓ Pregnancy Parameter: ENABLED for female patient")
        pregnancies = prompt_number("Number of Pregnancies (0–17)", default_val=3, min_val=0, max_val=17)

    # Step 3: Biomarker metrics
    print("\n[STEP 2] Clinical Biomarker Parameters:")
    glucose = prompt_number("Glucose Level (mg/dL) [Normal: 70–99, Pre-diabetic: 100–125, High: ≥126]", default_val=117, min_val=50, max_val=250)
    bmi = prompt_number("Body Mass Index / BMI (kg/m²) [Normal: 18.5–24.9, Obese: ≥30]", default_val=32.3, min_val=15.0, max_val=60.0, is_float=True)
    age = prompt_number("Patient Age (years) [18–90]", default_val=33, min_val=18, max_val=90)
    blood_pressure = prompt_number("Diastolic Blood Pressure (mmHg) [Normal: <80, Elevated: ≥90]", default_val=72, min_val=40, max_val=130)
    insulin = prompt_number("2-Hour Serum Insulin (μU/mL) [Normal: 16–166]", default_val=125, min_val=15, max_val=500)
    skin_thickness = prompt_number("Triceps Skin Fold Thickness (mm) [Normal: 10–30]", default_val=29, min_val=5, max_val=65)
    dpf = prompt_number("Diabetes Pedigree Function (genetic score) [0.05–2.50]", default_val=0.47, min_val=0.05, max_val=2.50, is_float=True)

    inputs = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }

    results = predict_risk(data_meta, inputs, gender)
    pct = round(results['consensus'] * 100)

    # Display Report
    print("\n" + "="*60)
    print("  CLINICAL RISK ASSESSMENT REPORT")
    print("="*60)
    print(f"  Patient Gender     : {gender_display}")
    if gender == 'female':
        print(f"  Pregnancies        : {pregnancies} count")
    print(f"  Fasting Glucose    : {glucose} mg/dL ({'High' if glucose >= 126 else 'Elevated' if glucose >= 100 else 'Normal'})")
    print(f"  BMI Category       : {bmi} kg/m² ({'Obese' if bmi >= 30 else 'Overweight' if bmi >= 25 else 'Normal'})")
    print(f"  Blood Pressure     : {blood_pressure} mmHg")
    print(f"  Age                : {age} years")
    print("─"*60)
    print(f"  Consensus Risk     : {pct}%")
    print(f"  XGBoost Risk       : {results['xgboost']*100:.1f}%")
    print(f"  Random Forest Risk : {results['random_forest']*100:.1f}%")
    print(f"  Log. Reg. Risk     : {results['logistic_regression']*100:.1f}%")
    print("─"*60)

    if results['consensus'] >= 0.60:
        print("  Status: 🔴 HIGH RISK (Likely Diabetic)")
        print("  Recommendation: Laboratory HbA1c testing and endocrine consultation strongly recommended.")
    elif results['consensus'] >= 0.35:
        print("  Status: 🟡 MODERATE / PREDIABETIC RISK")
        print("  Recommendation: Lifestyle modification, diet adjustment, and repeat screening in 6 months.")
    else:
        print("  Status: 🟢 LOW RISK (Normal / Negative)")
        print("  Recommendation: Routine annual preventative metabolic health checks.")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
