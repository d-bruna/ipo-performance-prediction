"""
Train all models and generate results.
"""

import sys
import os
sys.path.append('src')

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import cross_val_predict

from config import *
from models import prepare_data, train_models, compute_roc_auc, get_feature_importance


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def plot_roc_curves(models_1day, models_1year, X_1day, y_1day, X_1year, y_1year, cv):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    colors = {'dt': 'darkorange', 'rf': 'darkgreen', 'gb': 'darkblue', 'log': 'darkred'}
    
    for key, name in MODEL_NAMES.items():
        y_proba = cross_val_predict(models_1day[key], X_1day, y_1day, cv=cv, method='predict_proba')
        fpr, tpr, _ = roc_curve(y_1day, y_proba[:, 1])
        auc_score = roc_auc_score(y_1day, y_proba[:, 1])
        ax1.plot(fpr, tpr, color=colors[key], lw=3, label=f'{name} (AUC = {auc_score:.3f})')
    
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    ax1.set_xlabel('False Positive Rate', fontsize=12)
    ax1.set_ylabel('True Positive Rate', fontsize=12)
    ax1.set_title('1-Day Excess Return Prediction', fontsize=14)
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)
    
    for key, name in MODEL_NAMES.items():
        y_proba = cross_val_predict(models_1year[key], X_1year, y_1year, cv=cv, method='predict_proba')
        fpr, tpr, _ = roc_curve(y_1year, y_proba[:, 1])
        auc_score = roc_auc_score(y_1year, y_proba[:, 1])
        ax2.plot(fpr, tpr, color=colors[key], lw=3, label=f'{name} (AUC = {auc_score:.3f})')
    
    ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    ax2.set_xlabel('False Positive Rate', fontsize=12)
    ax2.set_ylabel('True Positive Rate', fontsize=12)
    ax2.set_title('1-Year Excess Return Prediction', fontsize=14)
    ax2.legend(loc="lower right")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{FIGURES_PATH}roc_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved: {FIGURES_PATH}roc_comparison.png")
    plt.close()


def main():
    print("="*70)
    print("IPO PERFORMANCE PREDICTION - MODEL TRAINING")
    print("="*70)
    
    ensure_dir(RESULTS_PATH)
    ensure_dir(FIGURES_PATH)
    ensure_dir(TABLES_PATH)
    ensure_dir(MODELS_PATH)
    
    print(f"\n[1/5] Loading data...")
    df = pd.read_csv(DATA_PATH)
    print(f"      Loaded {len(df)} IPOs")
    
    print("\n[2/5] Preparing data...")
    X_1day, y_1day, X_1year, y_1year = prepare_data(df, FEATURE_COLS, TARGET_1DAY_COL, TARGET_1YEAR_COL)
    
    print("\n[3/5] Training 1-day models...")
    models_1day, results_1day, cv = train_models(X_1day, y_1day, "1-DAY", use_smote=True)
    
    print("\n[4/5] Training 1-year models...")
    models_1year, results_1year, _ = train_models(X_1year, y_1year, "1-YEAR", use_smote=False)
    
    print("\n[5/5] Computing metrics and saving...")
    roc_1day = compute_roc_auc(models_1day, X_1day, y_1day, cv)
    roc_1year = compute_roc_auc(models_1year, X_1year, y_1year, cv)
    
    performance_1day = pd.DataFrame({
        'Model': list(results_1day.keys()),
        'Accuracy': list(results_1day.values()),
        'ROC_AUC': [roc_1day[k] for k in results_1day.keys()]
    }).sort_values('Accuracy', ascending=False)
    
    performance_1year = pd.DataFrame({
        'Model': list(results_1year.keys()),
        'Accuracy': list(results_1year.values()),
        'ROC_AUC': [roc_1year[k] for k in results_1year.keys()]
    }).sort_values('Accuracy', ascending=False)
    
    performance_1day.to_csv(f'{TABLES_PATH}performance_1day.csv', index=False)
    performance_1year.to_csv(f'{TABLES_PATH}performance_1year.csv', index=False)
    
    plot_roc_curves(models_1day, models_1year, X_1day, y_1day, X_1year, y_1year, cv)
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print("\n1-DAY:")
    print(performance_1day.to_string(index=False))
    print("\n1-YEAR:")
    print(performance_1year.to_string(index=False))
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
