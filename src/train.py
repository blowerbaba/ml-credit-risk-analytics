import joblib
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.config import BEST_MODEL_PATH, METRICS_PATH
from src.data_loader import load_or_create_data
from src.feature_engineering import FeatureEngineer
from src.evaluate import compute_model_metrics, calculate_shap_feature_importance, save_benchmark_metrics

def train_and_benchmark_models() -> dict:
    """
    Trains multiple ML classification models (Logistic Regression, Random Forest,
    XGBoost, LightGBM), benchmarks performance, selects the best model,
    saves model artifacts and calculates SHAP values.
    """
    print("Step 1: Loading Dataset...")
    df = load_or_create_data()
    
    print("Step 2: Feature Engineering & Preprocessing...")
    fe = FeatureEngineer()
    X, y = fe.fit_transform(df)
    
    # Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # Model Zoo
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight="balanced"),
        "XGBoost": XGBClassifier(n_estimators=120, max_depth=5, learning_rate=0.08, random_state=42, eval_metric="logloss"),
        "LightGBM": LGBMClassifier(n_estimators=120, max_depth=5, learning_rate=0.08, random_state=42, verbose=-1)
    }
    
    benchmark_results = {}
    best_model = None
    best_model_name = ""
    best_roc_auc = 0.0
    
    print("Step 3: Training & Benchmarking Candidate Models...")
    for name, model in models.items():
        print(f"   -> Training {name}...")
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        metrics = compute_model_metrics(y_test, y_pred, y_prob)
        benchmark_results[name] = metrics
        print(f"      [{name}] ROC-AUC: {metrics['roc_auc']} | PR-AUC: {metrics['pr_auc']} | F1-Score: {metrics['f1_score']}")
        
        if metrics["roc_auc"] > best_roc_auc:
            best_roc_auc = metrics["roc_auc"]
            best_model = model
            best_model_name = name

    print(f"\n[BEST MODEL SELECTED] {best_model_name} with ROC-AUC = {best_roc_auc}")
    
    # Save Best Model Artifact
    joblib.dump(best_model, BEST_MODEL_PATH)
    print(f"Model saved to {BEST_MODEL_PATH}")
    
    # Calculate SHAP feature importances on test sample
    print("Step 4: Computing SHAP Feature Importances...")
    X_shap_sample = X_test.sample(n=min(300, len(X_test)), random_state=42)
    top_shap = calculate_shap_feature_importance(best_model, X_shap_sample)
    
    # Compile Full Benchmarks Output
    final_output = {
        "best_model_name": best_model_name,
        "best_roc_auc": best_roc_auc,
        "models": benchmark_results,
        "feature_importances": top_shap
    }
    
    save_benchmark_metrics(final_output)
    print(f"Benchmark results saved to {METRICS_PATH}")
    
    return final_output

if __name__ == "__main__":
    train_and_benchmark_models()
