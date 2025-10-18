"""
Model training and evaluation for IPO performance prediction.
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import classification_report, roc_curve, roc_auc_score, accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings

warnings.filterwarnings('ignore')


def prepare_data(df, feature_cols, target_1day_col, target_1year_col):
    """Prepare feature matrices and target vectors."""
    if '1_day_excess_up' not in df.columns:
        df['1_day_excess_up'] = (df[target_1day_col] > 0).astype(int)
    if '1_year_excess_up' not in df.columns:
        df['1_year_excess_up'] = (df[target_1year_col] > 0).astype(int)
    
    X_1day = df[feature_cols].copy()
    y_1day = df['1_day_excess_up'].copy()
    valid_1day = ~y_1day.isnull()
    X_1day = X_1day[valid_1day].fillna(X_1day[valid_1day].median())
    y_1day = y_1day[valid_1day]
    
    X_1year = df[feature_cols].copy()
    y_1year = df['1_year_excess_up'].copy()
    valid_1year = ~y_1year.isnull()
    X_1year = X_1year[valid_1year].fillna(X_1year[valid_1year].median())
    y_1year = y_1year[valid_1year]
    
    print(f"\nData prepared:")
    print(f"  1-Day: {X_1day.shape[0]} samples, {X_1day.shape[1]} features")
    print(f"  1-Year: {X_1year.shape[0]} samples, {X_1year.shape[1]} features")
    
    return X_1day, y_1day, X_1year, y_1year


def train_models(X, y, target_name, use_smote=True, cv_folds=10):
    """Train all models with hyperparameter tuning."""
    print(f"\n{'='*70}")
    print(f"TRAINING MODELS FOR {target_name}")
    print(f"{'='*70}")
    
    if use_smote:
        print("Note: SMOTE will be applied for class imbalance.")
    else:
        print("Note: Using balanced class weights.")
    
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    param_grid_dt = {
        'classifier__criterion': ['gini', 'entropy'],
        'classifier__max_depth': [3, 5, 7],
        'classifier__min_samples_split': [10, 15],
        'classifier__min_samples_leaf': [10, 15]
    }
    
    param_grid_rf = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [3, 5, 7],
        'classifier__min_samples_split': [10, 15],
        'classifier__min_samples_leaf': [10, 15],
        'classifier__max_features': ['sqrt']
    }
    
    param_grid_gb = {
        'classifier__n_estimators': [50, 100],
        'classifier__max_depth': [3, 5],
        'classifier__learning_rate': [0.05, 0.1],
        'classifier__min_samples_split': [10, 15],
        'classifier__min_samples_leaf': [10],
        'classifier__subsample': [0.8]
    }
    
    param_grid_log = {
        'classifier__C': [0.1, 1.0, 10.0],
        'classifier__l1_ratio': [0.0, 0.5, 1.0],
        'classifier__class_weight': [None, 'balanced']
    }
    
    models = {}
    results = {}
    
    print("\n[1/4] Training Decision Tree...")
    if use_smote:
        dt_pipeline = ImbPipeline([
            ('smote', SMOTE(random_state=42)),
            ('classifier', DecisionTreeClassifier(random_state=42))
        ])
    else:
        dt_pipeline = Pipeline([
            ('classifier', DecisionTreeClassifier(random_state=42, class_weight='balanced'))
        ])
    
    grid_dt = GridSearchCV(dt_pipeline, param_grid_dt, cv=cv, scoring='accuracy', n_jobs=-1, verbose=0)
    grid_dt.fit(X, y)
    models['dt'] = grid_dt.best_estimator_
    results['Decision Tree'] = grid_dt.best_score_
    print(f"      Best accuracy: {grid_dt.best_score_:.4f}")
    
    print("[2/4] Training Random Forest...")
    if use_smote:
        rf_pipeline = ImbPipeline([
            ('smote', SMOTE(random_state=42)),
            ('classifier', RandomForestClassifier(random_state=42, n_jobs=-1))
        ])
    else:
        rf_pipeline = Pipeline([
            ('classifier', RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced'))
        ])
    
    grid_rf = GridSearchCV(rf_pipeline, param_grid_rf, cv=cv, scoring='accuracy', n_jobs=1, verbose=0)
    grid_rf.fit(X, y)
    models['rf'] = grid_rf.best_estimator_
    results['Random Forest'] = grid_rf.best_score_
    print(f"      Best accuracy: {grid_rf.best_score_:.4f}")
    
    print("[3/4] Training Gradient Boosting...")
    if use_smote:
        gb_pipeline = ImbPipeline([
            ('smote', SMOTE(random_state=42)),
            ('classifier', GradientBoostingClassifier(random_state=42))
        ])
    else:
        gb_pipeline = Pipeline([
            ('classifier', GradientBoostingClassifier(random_state=42))
        ])
    
    grid_gb = GridSearchCV(gb_pipeline, param_grid_gb, cv=cv, scoring='accuracy', n_jobs=-1, verbose=0)
    grid_gb.fit(X, y)
    models['gb'] = grid_gb.best_estimator_
    results['Gradient Boosting'] = grid_gb.best_score_
    print(f"      Best accuracy: {grid_gb.best_score_:.4f}")
    
    print("[4/4] Training Elastic-Net Logistic...")
    if use_smote:
        log_pipeline = ImbPipeline([
            ('smote', SMOTE(random_state=42)),
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(penalty='elasticnet', solver='saga', max_iter=10000, random_state=42))
        ])
    else:
        log_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(penalty='elasticnet', solver='saga', max_iter=10000, random_state=42, class_weight='balanced'))
        ])
    
    grid_log = GridSearchCV(log_pipeline, param_grid_log, cv=cv, scoring='accuracy', n_jobs=-1, verbose=0)
    grid_log.fit(X, y)
    models['log'] = grid_log.best_estimator_
    results['Elastic-Net Logistic'] = grid_log.best_score_
    print(f"      Best accuracy: {grid_log.best_score_:.4f}")
    
    print("\nTraining complete.")
    return models, results, cv


def compute_roc_auc(models, X, y, cv):
    """Compute ROC AUC scores."""
    from config import MODEL_NAMES
    roc_scores = {}
    
    for key, name in MODEL_NAMES.items():
        y_proba = cross_val_predict(models[key], X, y, cv=cv, method='predict_proba')
        roc_scores[name] = roc_auc_score(y, y_proba[:, 1])
    
    return roc_scores


def get_feature_importance(model, feature_cols):
    """Extract feature importance."""
    if hasattr(model.named_steps['classifier'], 'feature_importances_'):
        importances = model.named_steps['classifier'].feature_importances_
        return pd.DataFrame({
            'Feature': feature_cols,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
    return None
