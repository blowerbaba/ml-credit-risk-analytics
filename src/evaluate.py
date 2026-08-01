import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix
)
import shap
from src.config import METRICS_PATH, SHAP_SUMMARY_PATH

def compute_model_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """Computes comprehensive evaluation metrics for classification models."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    metrics = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_prob)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred)), 4),
        "recall": round(float(recall_score(y_true, y_pred)), 4),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        }
    }
    return metrics

def calculate_shap_feature_importance(model, X_sample: pd.DataFrame, num_features: int = 10) -> dict:
    """
    Calculates SHAP global feature importances for tree-based or linear models.
    Returns a dictionary of feature names and mean absolute SHAP values.
    """
    try:
        if hasattr(model, "predict_proba") and ("Tree" in type(model).__name__ or "XGB" in type(model).__name__ or "LGBM" in type(model).__name__):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # positive class
        else:
            explainer = shap.Explainer(model, X_sample)
            shap_values = explainer(X_sample).values
            if len(shap_values.shape) == 3:
                shap_values = shap_values[:, :, 1]

        mean_shap = np.abs(shap_values).mean(axis=0)
        feature_importance = pd.Series(mean_shap, index=X_sample.columns).sort_values(ascending=False)
        
        top_features = feature_importance.head(num_features).to_dict()
        top_features_clean = {str(k): round(float(v), 4) for k, v in top_features.items()}
        
        with open(SHAP_SUMMARY_PATH, "w") as f:
            json.dump(top_features_clean, f, indent=4)
            
        return top_features_clean
    except Exception as e:
        print(f"Warning: SHAP calculation fallback triggered due to: {e}")
        # Fallback to feature_importances_ or coef_
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_[0])
        else:
            importances = np.ones(X_sample.shape[1])
            
        feature_importance = pd.Series(importances, index=X_sample.columns).sort_values(ascending=False)
        top_features = {str(k): round(float(v), 4) for k, v in feature_importance.head(num_features).items()}
        
        with open(SHAP_SUMMARY_PATH, "w") as f:
            json.dump(top_features, f, indent=4)
            
        return top_features

def save_benchmark_metrics(metrics_dict: dict):
    """Saves benchmark results to JSON artifact."""
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_dict, f, indent=4)
