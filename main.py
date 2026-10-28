"""
=============================================================
  DIABETES PREDICTION - DATA SCIENCE PROJECT
  Using: Logistic Regression, Random Forest, XGBoost
=============================================================
  Dataset : Pima Indians Diabetes Database
  Author  : Data Science Pipeline
  Date    : 2026-10-19
=============================================================
"""

# ── Imports ──────────────────────────────────────────────────
import os
import sys
import json
import shutil
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Fix Windows console encoding for Unicode characters
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

# ── Configuration ────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
PLOTS_DIR = os.path.join(os.path.dirname(__file__), 'plots')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
DASHBOARD_DIR = os.path.join(os.path.dirname(__file__), 'dashboard')

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

np.random.seed(RANDOM_STATE)

# ── Color Palette ────────────────────────────────────────────
COLORS = {
    'primary': '#6C5CE7',
    'secondary': '#00CEC9',
    'accent': '#FD79A8',
    'warning': '#FDCB6E',
    'dark': '#2D3436',
    'light': '#DFE6E9',
    'positive': '#00B894',
    'negative': '#D63031',
}

plt.rcParams.update({
    'figure.facecolor': '#1a1a2e',
    'axes.facecolor': '#16213e',
    'axes.edgecolor': '#e94560',
    'axes.labelcolor': '#eaeaea',
    'text.color': '#eaeaea',
    'xtick.color': '#eaeaea',
    'ytick.color': '#eaeaea',
    'grid.color': '#2a2a4a',
    'font.size': 11,
    'font.family': 'sans-serif',
})


# =============================================================
#  STEP 1: DATA LOADING
# =============================================================
def load_data():
    """
    Load the Pima Indians Diabetes Dataset.
    Downloads from UCI/Kaggle mirror if not available locally.
    """
    print("\n" + "="*60)
    print("  STEP 1: LOADING DATA")
    print("="*60)

    csv_path = os.path.join(DATA_DIR, 'diabetes.csv')

    # Try loading from local file first
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"  ✓ Loaded dataset from local file: {csv_path}")
    else:
        # Download from public URL
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
        columns = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
        ]
        try:
            df = pd.read_csv(url, header=None, names=columns)
            df.to_csv(csv_path, index=False)
            print(f"  ✓ Downloaded dataset and saved to: {csv_path}")
        except Exception as e:
            print(f"  ⚠ Could not download dataset: {e}")
            print("  → Generating synthetic Pima-like dataset...")
            df = _generate_synthetic_dataset()
            df.to_csv(csv_path, index=False)
            print(f"  ✓ Synthetic dataset saved to: {csv_path}")

    print(f"  → Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"  → Columns: {list(df.columns)}")
    return df


def _generate_synthetic_dataset(n_samples=768):
    """Generate a realistic synthetic Pima Indians-like dataset."""
    rng = np.random.RandomState(RANDOM_STATE)

    # Generate features with realistic distributions
    pregnancies = rng.poisson(3.8, n_samples).clip(0, 17)
    glucose = rng.normal(121, 32, n_samples).clip(0, 199).astype(int)
    blood_pressure = rng.normal(69, 19, n_samples).clip(0, 122).astype(int)
    skin_thickness = rng.normal(20, 16, n_samples).clip(0, 99).astype(int)
    insulin = rng.normal(80, 115, n_samples).clip(0, 846).astype(int)
    bmi = rng.normal(32, 8, n_samples).clip(0, 67.1).round(1)
    dpf = rng.exponential(0.47, n_samples).clip(0.078, 2.42).round(3)
    age = rng.normal(33, 12, n_samples).clip(21, 81).astype(int)

    # Generate outcome with realistic probability based on features
    log_odds = (
        -8.0
        + 0.12 * pregnancies
        + 0.035 * glucose
        + 0.01 * bmi
        + 0.5 * dpf
        + 0.015 * age
    )
    prob = 1 / (1 + np.exp(-log_odds))
    outcome = (rng.random(n_samples) < prob).astype(int)

    df = pd.DataFrame({
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age,
        'Outcome': outcome,
    })
    return df


# =============================================================
#  STEP 2: EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================
def perform_eda(df):
    """Comprehensive Exploratory Data Analysis."""
    print("\n" + "="*60)
    print("  STEP 2: EXPLORATORY DATA ANALYSIS")
    print("="*60)

    # --- 2.1 Basic Info ---
    print("\n── 2.1 Dataset Overview ──")
    print(f"  Shape            : {df.shape}")
    print(f"  Duplicated rows  : {df.duplicated().sum()}")
    print(f"  Missing values   : {df.isnull().sum().sum()}")

    print("\n── 2.2 Statistical Summary ──")
    desc = df.describe().round(2)
    print(desc.to_string())

    print("\n── 2.3 Target Distribution ──")
    outcome_counts = df['Outcome'].value_counts()
    total = len(df)
    for label, count in outcome_counts.items():
        tag = "Non-Diabetic" if label == 0 else "Diabetic"
        print(f"  {tag} (class {label}): {count}  ({count/total*100:.1f}%)")

    # --- 2.4 Zero-value Analysis ---
    print("\n── 2.4 Zero-Value Analysis ──")
    zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    zero_stats = {}
    for col in zero_cols:
        if col in df.columns:
            n_zero = (df[col] == 0).sum()
            pct = n_zero / total * 100
            zero_stats[col] = {'count': int(n_zero), 'percentage': round(pct, 2)}
            print(f"  {col:30s} → {n_zero:4d} zeros ({pct:.1f}%)")

    # --- 2.5 Correlation Analysis ---
    print("\n── 2.5 Top Correlations with Outcome ──")
    corr_with_outcome = df.corr()['Outcome'].drop('Outcome').sort_values(ascending=False)
    for feat, corr in corr_with_outcome.items():
        bar = '█' * int(abs(corr) * 40)
        sign = '+' if corr > 0 else '-'
        print(f"  {feat:30s}  {sign}{abs(corr):.4f}  {bar}")

    # --- Generate EDA Plots ---
    _plot_target_distribution(df)
    _plot_feature_distributions(df)
    _plot_correlation_heatmap(df)
    _plot_feature_boxplots(df)
    _plot_pairplot_key_features(df)

    # Build EDA summary for dashboard
    eda_summary = {
        'shape': list(df.shape),
        'missing_values': int(df.isnull().sum().sum()),
        'duplicated_rows': int(df.duplicated().sum()),
        'target_distribution': {
            'non_diabetic': int(outcome_counts.get(0, 0)),
            'diabetic': int(outcome_counts.get(1, 0)),
        },
        'zero_value_analysis': zero_stats,
        'correlations': {k: round(v, 4) for k, v in corr_with_outcome.items()},
        'statistics': json.loads(desc.to_json()),
    }
    return eda_summary


def _plot_target_distribution(df):
    """Plot the target variable distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Pie chart
    counts = df['Outcome'].value_counts()
    colors_pie = [COLORS['positive'], COLORS['negative']]
    axes[0].pie(
        counts, labels=['Non-Diabetic', 'Diabetic'], autopct='%1.1f%%',
        colors=colors_pie, startangle=90, textprops={'fontsize': 13, 'fontweight': 'bold'},
        wedgeprops={'edgecolor': '#1a1a2e', 'linewidth': 2}
    )
    axes[0].set_title('Target Distribution (Pie)', fontsize=14, fontweight='bold', pad=15)

    # Bar chart
    bars = axes[1].bar(['Non-Diabetic\n(Class 0)', 'Diabetic\n(Class 1)'],
                       counts.values, color=colors_pie, edgecolor='#1a1a2e', linewidth=2)
    for bar, count in zip(bars, counts.values):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 8,
                     str(count), ha='center', fontweight='bold', fontsize=13)
    axes[1].set_title('Target Distribution (Bar)', fontsize=14, fontweight='bold', pad=15)
    axes[1].set_ylabel('Count')
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '01_target_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: 01_target_distribution.png")


def _plot_feature_distributions(df):
    """Plot histograms for all features."""
    features = df.columns.drop('Outcome')
    n_features = len(features)
    cols = 4
    rows = (n_features + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 4))
    axes = axes.flatten()

    for i, feat in enumerate(features):
        ax = axes[i]
        ax.hist(df[df['Outcome'] == 0][feat], bins=30, alpha=0.6,
                color=COLORS['primary'], label='Non-Diabetic', edgecolor='none')
        ax.hist(df[df['Outcome'] == 1][feat], bins=30, alpha=0.6,
                color=COLORS['accent'], label='Diabetic', edgecolor='none')
        ax.set_title(feat, fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.2)

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Feature Distributions by Outcome', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '02_feature_distributions.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: 02_feature_distributions.png")


def _plot_correlation_heatmap(df):
    """Plot correlation heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))
    corr_matrix = df.corr()

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    cmap = sns.diverging_palette(250, 15, s=75, l=40, n=9, center='dark', as_cmap=True)

    sns.heatmap(
        corr_matrix, mask=mask, annot=True, fmt='.2f', cmap=cmap,
        center=0, square=True, linewidths=1, linecolor='#1a1a2e',
        cbar_kws={'shrink': 0.8}, ax=ax,
        annot_kws={'fontsize': 10, 'fontweight': 'bold'}
    )
    ax.set_title('Feature Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '03_correlation_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: 03_correlation_heatmap.png")


def _plot_feature_boxplots(df):
    """Plot boxplots grouped by outcome."""
    features = df.columns.drop('Outcome')
    n_features = len(features)
    cols = 4
    rows = (n_features + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 4))
    axes = axes.flatten()

    for i, feat in enumerate(features):
        ax = axes[i]
        data_0 = df[df['Outcome'] == 0][feat]
        data_1 = df[df['Outcome'] == 1][feat]

        bp = ax.boxplot(
            [data_0, data_1], labels=['Non-Diabetic', 'Diabetic'],
            patch_artist=True, widths=0.5,
            medianprops={'color': COLORS['warning'], 'linewidth': 2},
        )
        bp['boxes'][0].set_facecolor(COLORS['primary'])
        bp['boxes'][0].set_alpha(0.7)
        bp['boxes'][1].set_facecolor(COLORS['accent'])
        bp['boxes'][1].set_alpha(0.7)

        ax.set_title(feat, fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.2)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Feature Box Plots by Outcome', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '04_feature_boxplots.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: 04_feature_boxplots.png")


def _plot_pairplot_key_features(df):
    """Scatter matrix of the top correlated features."""
    key_features = ['Glucose', 'BMI', 'Age', 'Pregnancies', 'Outcome']
    available = [f for f in key_features if f in df.columns]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    pairs = [
        ('Glucose', 'BMI'), ('Glucose', 'Age'), ('BMI', 'Age'),
        ('Pregnancies', 'Glucose'), ('Pregnancies', 'BMI'), ('Pregnancies', 'Age'),
    ]

    for idx, (x_feat, y_feat) in enumerate(pairs):
        if x_feat not in df.columns or y_feat not in df.columns:
            continue
        ax = axes[idx // 3][idx % 3]
        for outcome, color, label in [(0, COLORS['primary'], 'Non-Diabetic'),
                                       (1, COLORS['accent'], 'Diabetic')]:
            subset = df[df['Outcome'] == outcome]
            ax.scatter(subset[x_feat], subset[y_feat], c=color, alpha=0.5,
                       s=20, label=label, edgecolors='none')
        ax.set_xlabel(x_feat, fontsize=10)
        ax.set_ylabel(y_feat, fontsize=10)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(alpha=0.2)

    fig.suptitle('Feature Pair Plots (Key Features)', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '05_pair_plots.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: 05_pair_plots.png")


# =============================================================
#  STEP 3: DATA PREPROCESSING
# =============================================================
def preprocess_data(df):
    """Clean and preprocess the dataset."""
    print("\n" + "="*60)
    print("  STEP 3: DATA PREPROCESSING")
    print("="*60)

    df_clean = df.copy()

    # 3.1 Replace zeros with NaN for columns where 0 is not physiologically possible
    zero_replace_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    existing_cols = [c for c in zero_replace_cols if c in df_clean.columns]

    print("\n── 3.1 Replacing biologically impossible zeros with NaN ──")
    for col in existing_cols:
        n_zeros = (df_clean[col] == 0).sum()
        if n_zeros > 0:
            df_clean[col] = df_clean[col].replace(0, np.nan)
            print(f"  {col:30s} → replaced {n_zeros} zeros")

    # 3.2 Impute missing values with median (robust to outliers)
    print("\n── 3.2 Imputing missing values with median ──")
    for col in existing_cols:
        n_missing = df_clean[col].isnull().sum()
        if n_missing > 0:
            median_val = float(df_clean[col].median())
            df_clean[col] = df_clean[col].fillna(median_val)
            print(f"  {col:30s} → filled {n_missing} NaN with median = {median_val:.2f}")

    # 3.3 Feature Engineering
    print("\n── 3.3 Feature Engineering ──")
    if 'Glucose' in df_clean.columns and 'Insulin' in df_clean.columns:
        df_clean['Glucose_Insulin_Ratio'] = (
            df_clean['Glucose'] / (df_clean['Insulin'] + 1)
        ).round(4)
        print("  ✓ Created: Glucose_Insulin_Ratio")

    if 'BMI' in df_clean.columns and 'Age' in df_clean.columns:
        df_clean['BMI_Age_Product'] = (df_clean['BMI'] * df_clean['Age']).round(2)
        print("  ✓ Created: BMI_Age_Product")

    if 'Age' in df_clean.columns:
        df_clean['Age_Group'] = pd.cut(
            df_clean['Age'], bins=[-np.inf, 25, 35, 45, 55, np.inf],
            labels=[0, 1, 2, 3, 4]
        ).astype(int)
        print("  ✓ Created: Age_Group (0: ≤25, 1: 26-35, 2: 36-45, 3: 46-55, 4: 56+)")

    # Ensure no remaining NaNs or infinite values across all columns
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
    for col in df_clean.columns:
        if df_clean[col].isnull().sum() > 0:
            col_median = float(df_clean[col].median())
            df_clean[col] = df_clean[col].fillna(col_median)

    # 3.4 Split features and target
    X = df_clean.drop('Outcome', axis=1)
    y = df_clean['Outcome']

    # 3.5 Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\n── 3.4 Train/Test Split ──")
    print(f"  Training set   : {X_train.shape[0]} samples")
    print(f"  Test set       : {X_test.shape[0]} samples")
    print(f"  Features       : {X_train.shape[1]}")

    # 3.5 Feature Scaling
    scaler = StandardScaler()
    feature_names = X_train.columns.tolist()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_names, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_names, index=X_test.index)
    print(f"\n── 3.5 Feature Scaling ──")
    print(f"  ✓ Applied StandardScaler (mean=0, std=1)")

    preprocessing_info = {
        'n_features': X_train.shape[1],
        'feature_names': feature_names,
        'train_size': X_train.shape[0],
        'test_size': X_test.shape[0],
        'engineered_features': ['Glucose_Insulin_Ratio', 'BMI_Age_Product', 'Age_Group'],
        'scaler_mean': {feat: round(float(m), 4) for feat, m in zip(feature_names, scaler.mean_)},
        'scaler_scale': {feat: round(float(s), 4) for feat, s in zip(feature_names, scaler.scale_)},
    }

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, preprocessing_info


# =============================================================
#  STEP 4: MODEL TRAINING & EVALUATION
# =============================================================
def train_and_evaluate(X_train, X_test, y_train, y_test):
    """Train Logistic Regression, Random Forest, and XGBoost models."""
    print("\n" + "="*60)
    print("  STEP 4: MODEL TRAINING & EVALUATION")
    print("="*60)

    models = {
        'Logistic Regression': LogisticRegression(
            C=1.0, max_iter=1000, random_state=RANDOM_STATE, solver='lbfgs'
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_split=5,
            min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1
        ),
        'XGBoost': XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_STATE, eval_metric='logloss',
            use_label_encoder=False
        ),
    }

    results = {}
    all_roc_data = {}

    for name, model in models.items():
        print(f"\n{'─'*50}")
        print(f"  Training: {name}")
        print(f"{'─'*50}")

        # Train
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)

        # Cross-Validation
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)

        # ROC Curve data
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        all_roc_data[name] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'auc': roc_auc}

        # Feature Importance
        feature_importance = _get_feature_importance(model, X_train.columns.tolist(), name)

        # Print results
        print(f"  Accuracy      : {accuracy:.4f}")
        print(f"  Precision     : {precision:.4f}")
        print(f"  Recall        : {recall:.4f}")
        print(f"  F1 Score      : {f1:.4f}")
        print(f"  ROC-AUC       : {roc_auc:.4f}")
        print(f"  CV Accuracy   : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"\n  Confusion Matrix:")
        print(f"    TN={cm[0][0]:4d}  FP={cm[0][1]:4d}")
        print(f"    FN={cm[1][0]:4d}  TP={cm[1][1]:4d}")

        report = classification_report(y_test, y_pred, output_dict=True)

        results[name] = {
            'accuracy': round(accuracy, 4),
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1_score': round(f1, 4),
            'roc_auc': round(roc_auc, 4),
            'cv_mean': round(cv_scores.mean(), 4),
            'cv_std': round(cv_scores.std(), 4),
            'cv_scores': [round(s, 4) for s in cv_scores.tolist()],
            'confusion_matrix': cm.tolist(),
            'classification_report': {
                k: {kk: round(vv, 4) for kk, vv in v.items()} if isinstance(v, dict) else round(v, 4)
                for k, v in report.items()
            },
            'feature_importance': feature_importance,
        }

        if name == 'Logistic Regression':
            results[name]['intercept'] = round(float(model.intercept_[0]), 4)
            results[name]['coefficients'] = {
                feat: round(float(c), 4) for feat, c in zip(X_train.columns, model.coef_[0])
            }

    # Generate comparison plots
    _plot_model_comparison(results)
    _plot_roc_curves(all_roc_data)
    _plot_confusion_matrices(results)
    _plot_feature_importance(results)

    return results, all_roc_data


def _get_feature_importance(model, feature_names, model_name):
    """Extract feature importance from the model."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        return {}

    # Normalize
    importances = importances / importances.sum()
    importance_dict = {name: round(float(imp), 4) for name, imp in zip(feature_names, importances)}
    return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))


def _plot_model_comparison(results):
    """Bar chart comparing model metrics."""
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    model_names = list(results.keys())
    n_metrics = len(metrics)

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(n_metrics)
    width = 0.25
    colors_models = [COLORS['primary'], COLORS['secondary'], COLORS['accent']]

    for i, model_name in enumerate(model_names):
        values = [results[model_name][m] for m in metrics]
        bars = ax.bar(x + i * width, values, width, label=model_name,
                      color=colors_models[i], edgecolor='#1a1a2e', linewidth=1.5)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xlabel('Metrics', fontsize=13, fontweight='bold')
    ax.set_ylabel('Score', fontsize=13, fontweight='bold')
    ax.set_title('Model Performance Comparison', fontsize=16, fontweight='bold', pad=15)
    ax.set_xticks(x + width)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics], fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '06_model_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: 06_model_comparison.png")


def _plot_roc_curves(roc_data):
    """Plot ROC curves for all models."""
    fig, ax = plt.subplots(figsize=(8, 8))
    colors_models = [COLORS['primary'], COLORS['secondary'], COLORS['accent']]

    for i, (name, data) in enumerate(roc_data.items()):
        ax.plot(data['fpr'], data['tpr'], color=colors_models[i],
                linewidth=2.5, label=f"{name} (AUC = {data['auc']:.4f})")

    ax.plot([0, 1], [0, 1], 'w--', alpha=0.4, linewidth=1)
    ax.set_xlabel('False Positive Rate', fontsize=13, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=13, fontweight='bold')
    ax.set_title('ROC Curves Comparison', fontsize=16, fontweight='bold', pad=15)
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(alpha=0.2)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '07_roc_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: 07_roc_curves.png")


def _plot_confusion_matrices(results):
    """Plot confusion matrices side by side."""
    model_names = list(results.keys())
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for i, name in enumerate(model_names):
        cm = np.array(results[name]['confusion_matrix'])
        ax = axes[i]

        sns.heatmap(
            cm, annot=True, fmt='d', cmap='magma', ax=ax,
            xticklabels=['Non-Diabetic', 'Diabetic'],
            yticklabels=['Non-Diabetic', 'Diabetic'],
            annot_kws={'fontsize': 16, 'fontweight': 'bold'},
            linewidths=2, linecolor='#1a1a2e',
        )
        ax.set_title(name, fontsize=13, fontweight='bold', pad=10)
        ax.set_xlabel('Predicted', fontsize=11)
        ax.set_ylabel('Actual', fontsize=11)

    fig.suptitle('Confusion Matrices', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '08_confusion_matrices.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: 08_confusion_matrices.png")


def _plot_feature_importance(results):
    """Plot feature importance for each model."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    colors_models = [COLORS['primary'], COLORS['secondary'], COLORS['accent']]
    model_names = list(results.keys())

    for i, name in enumerate(model_names):
        ax = axes[i]
        importance = results[name].get('feature_importance', {})
        if not importance:
            ax.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=14)
            ax.set_title(name, fontsize=13, fontweight='bold')
            continue

        features = list(importance.keys())[:10]  # Top 10
        values = [importance[f] for f in features]

        bars = ax.barh(features[::-1], values[::-1], color=colors_models[i],
                       edgecolor='#1a1a2e', linewidth=1)
        for bar, val in zip(bars, values[::-1]):
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                    f'{val:.3f}', va='center', fontsize=9, fontweight='bold')

        ax.set_title(name, fontsize=13, fontweight='bold', pad=10)
        ax.set_xlabel('Importance', fontsize=10)
        ax.grid(axis='x', alpha=0.2)

    fig.suptitle('Feature Importance (Top 10)', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '09_feature_importance.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: 09_feature_importance.png")


# =============================================================
#  STEP 5: SAVE RESULTS
# =============================================================
def save_results(eda_summary, model_results, roc_data, preprocessing_info):
    """Save all results as JSON for the dashboard."""
    print("\n" + "="*60)
    print("  STEP 5: SAVING RESULTS")
    print("="*60)

    # Determine best model
    best_model = max(model_results, key=lambda k: model_results[k]['roc_auc'])

    dashboard_data = {
        'project_title': 'Diabetes Prediction',
        'project_subtitle': 'Machine Learning Classification Project',
        'dataset': 'Pima Indians Diabetes Database',
        'date': '2026-10-19',
        'eda': eda_summary,
        'preprocessing': preprocessing_info,
        'models': model_results,
        'roc_data': roc_data,
        'best_model': {
            'name': best_model,
            'metrics': model_results[best_model],
        },
    }

    # Save to results directory
    results_path = os.path.join(RESULTS_DIR, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(dashboard_data, f, indent=2)
    print(f"  ✓ Saved results: {results_path}")

    # Also save to dashboard directory for the web app
    dashboard_json_path = os.path.join(DASHBOARD_DIR, 'results.json')
    with open(dashboard_json_path, 'w') as f:
        json.dump(dashboard_data, f, indent=2)
    print(f"  ✓ Saved dashboard data: {dashboard_json_path}")

    # Ensure dashboard/plots directory has the latest plots
    dashboard_plots = os.path.join(DASHBOARD_DIR, 'plots')
    os.makedirs(dashboard_plots, exist_ok=True)
    for plot_file in os.listdir(PLOTS_DIR):
        if plot_file.endswith('.png'):
            shutil.copy2(os.path.join(PLOTS_DIR, plot_file), os.path.join(dashboard_plots, plot_file))
    print(f"  ✓ Synced plots to: {dashboard_plots}")

    return dashboard_data


# =============================================================
#  STEP 6: FINAL SUMMARY
# =============================================================
def print_final_summary(results):
    """Print a beautiful final summary."""
    print("\n" + "="*60)
    print("  FINAL SUMMARY")
    print("="*60)

    best_model = max(results, key=lambda k: results[k]['roc_auc'])

    print(f"\n  {'Model':<25s} {'Accuracy':>10s} {'F1':>10s} {'ROC-AUC':>10s} {'CV Mean':>10s}")
    print(f"  {'─'*65}")
    for name, metrics in results.items():
        marker = ' ★' if name == best_model else ''
        print(f"  {name:<25s} {metrics['accuracy']:>10.4f} {metrics['f1_score']:>10.4f} "
              f"{metrics['roc_auc']:>10.4f} {metrics['cv_mean']:>10.4f}{marker}")

    print(f"\n  🏆 Best Model: {best_model}")
    print(f"     ROC-AUC: {results[best_model]['roc_auc']:.4f}")
    print(f"     F1 Score: {results[best_model]['f1_score']:.4f}")
    print(f"     CV Accuracy: {results[best_model]['cv_mean']:.4f} ± {results[best_model]['cv_std']:.4f}")

    print("\n  📊 Plots saved to: plots/")
    print("  📁 Results saved to: results/results.json")
    print("  🌐 Dashboard data saved to: dashboard/results.json")
    print("\n" + "="*60)
    print("  ✅ PROJECT COMPLETE!")
    print("="*60 + "\n")


# =============================================================
#  MAIN PIPELINE
# =============================================================
def main():
    """Execute the full Diabetes Prediction pipeline."""
    print("\n" + "╔" + "═"*58 + "╗")
    print("║" + " DIABETES PREDICTION - ML PIPELINE ".center(58) + "║")
    print("║" + " Logistic Regression | Random Forest | XGBoost ".center(58) + "║")
    print("╚" + "═"*58 + "╝")

    # Step 1: Load data
    df = load_data()

    # Step 2: EDA
    eda_summary = perform_eda(df)

    # Step 3: Preprocess
    X_train, X_test, y_train, y_test, scaler, prep_info = preprocess_data(df)

    # Step 4: Train & Evaluate
    model_results, roc_data = train_and_evaluate(X_train, X_test, y_train, y_test)

    # Step 5: Save results
    save_results(eda_summary, model_results, roc_data, prep_info)

    # Step 6: Final summary
    print_final_summary(model_results)


if __name__ == '__main__':
    main()
