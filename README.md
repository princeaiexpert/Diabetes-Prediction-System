# 🩺 Diabetes Prediction — Data Science Project

> Predict diabetes outcomes from health-related features using Machine Learning

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.2+-orange?style=flat-square)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-green?style=flat-square)

## 📋 Overview

This project builds and compares three machine learning models to predict diabetes using the **Pima Indians Diabetes Database**. The pipeline includes comprehensive EDA, data preprocessing, model training, evaluation, and an interactive web dashboard.

### Models Used
| Model | Description |
|-------|-------------|
| **Logistic Regression** | Linear classifier — baseline model |
| **Random Forest** | Ensemble of decision trees — captures non-linear patterns |
| **XGBoost** | Gradient-boosted trees — state-of-the-art performance |

### Dataset Features
| Feature | Description |
|---------|-------------|
| Pregnancies | Number of pregnancies |
| Glucose | Plasma glucose concentration (2h oral glucose tolerance test) |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Triceps skin fold thickness (mm) |
| Insulin | 2-hour serum insulin (μU/ml) |
| BMI | Body mass index (kg/m²) |
| DiabetesPedigreeFunction | Diabetes pedigree function score |
| Age | Age in years |
| **Outcome** | 0 = Non-diabetic, 1 = Diabetic (target) |

## 🏆 Model Performance Benchmark

Evaluation performed on the held-out stratified test set (154 samples) and 5-fold cross-validation (614 training samples):

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC | 5-Fold CV Accuracy |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **XGBoost (Best)** 🏆 | **76.62%** | **68.75%** | **61.11%** | **64.71%** | **0.8226** | 73.93% ± 3.04% |
| **Random Forest** | 75.32% | 65.38% | 62.96% | 64.15% | 0.8180 | 76.22% ± 3.47% |
| **Logistic Regression** | 68.83% | 56.82% | 46.30% | 51.02% | 0.8059 | **78.82% ± 2.18%** |

### Top Predictive Features (from XGBoost & Random Forest)
1. **Glucose Level** (~28% - 36% importance) — strongest individual predictor
2. **Body Mass Index (BMI)** (~16% importance)
3. **Age & BMI-Age Interaction** (~10% - 13% importance)
4. **Glucose-to-Insulin Ratio** (~9% importance)
5. **Diabetes Pedigree Function** (~8% importance)

---

## 🧪 Interactive Live Predictor

The web dashboard and CLI include a real-time clinical risk simulator:
- **Biological Gender Gating**: Interactive Male (👨) vs Female (👩) selection.
  - **Female**: Enables the Pregnancies parameter (0–17) with live fertility risk calibration.
  - **Male**: Automatically locks the Pregnancies parameter to 0 with disabled controls (`N/A for Male`).
- **Biomarker Controls**: Interactive sliders and synchronized numeric inputs for physiological indicators (Glucose, BMI, Age, Pregnancies, Blood Pressure, Insulin, Skin Thickness, Pedigree Function).
- **Clinical Reference Ranges**: Live badge indicators (Normal, Elevated, High) based on clinical thresholds (e.g. fasting glucose ≥126 mg/dL, BMI ≥30 kg/m²).
- **One-Click Patient Presets**: Pre-configured profiles for *Healthy Female*, *Healthy Male*, *Prediabetic*, *High Risk Female*, *High Risk Male*, and *Dataset Median*.
- **Multi-Model Predictions**: Toggle between exact Logistic Regression logit calculation, Random Forest heuristics, XGBoost non-linear decision boundaries, or a weighted Consensus Ensemble.
- **Dynamic Radial Risk Gauge**: Glowing SVG gauge displaying 0–100% risk with automated clinical triage recommendations.

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Machine Learning Pipeline
```bash
python main.py
```
This executes the full automated pipeline:
- Auto-downloads or loads the Pima Indians dataset
- Performs comprehensive exploratory data analysis (EDA)
- Imputes biological zeros using robust median replacement
- Engineers clinical interaction features (`Glucose_Insulin_Ratio`, `BMI_Age_Product`, `Age_Group`)
- Fits StandardScaler and trains Logistic Regression, Random Forest, and XGBoost
- Generates and saves 9 publication-grade matplotlib/seaborn plots
- Exports model parameters and synchronized results to `dashboard/results.json`

### 3. Launch the Interactive Dashboard
Open `dashboard/index.html` directly in any web browser, or launch via local web server:
```bash
python -m http.server 8080 --directory dashboard
```
Then visit `http://localhost:8080/index.html`.

---

## 📁 Project Structure

```
diabetes_prediction/
├── main.py                  # Full end-to-end ML pipeline
├── requirements.txt         # Python dependencies
├── README.md                # Comprehensive documentation
├── data/
│   └── diabetes.csv         # Pima Indians Diabetes dataset
├── plots/                   # Publication-quality visual analysis (9 plots)
│   ├── 01_target_distribution.png
│   ├── 02_feature_distributions.png
│   ├── 03_correlation_heatmap.png
│   ├── 04_feature_boxplots.png
│   ├── 05_pair_plots.png
│   ├── 06_model_comparison.png
│   ├── 07_roc_curves.png
│   ├── 08_confusion_matrices.png
│   └── 09_feature_importance.png
├── results/
│   └── results.json         # Serialized model metrics and coefficients
└── dashboard/
    ├── index.html           # Dark-mode dashboard with Live Clinical Predictor
    ├── results.json         # Synchronized results feed
    └── plots/               # Bundled visualization assets
```

## 🛠️ Technologies

- **Core ML**: Python 3.13, Scikit-learn, XGBoost, NumPy, Pandas
- **Visualization**: Matplotlib, Seaborn
- **Interactive UI**: HTML5, Modern CSS (Glassmorphism, CSS Custom Properties, SVG Radial Gauge, Micro-animations), Vanilla JavaScript

## 📄 License
This project is open-source and intended for educational and research purposes.
