import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from src.config import (
    BEST_MODEL_PATH,
    LOW_RISK_THRESHOLD,
    MEDIUM_RISK_THRESHOLD
)
from src.feature_engineering import FeatureEngineer

class CreditRiskPredictor:
    """
    Production inference engine for Credit Risk & Default Prediction.
    Loads trained artifacts, runs transformations, generates probabilities,
    categorizes risk levels, and identifies applicant risk factors.
    """
    def __init__(self):
        self.model = None
        self.feature_engineer = None
        self._load_artifacts()

    def _load_artifacts(self):
        """Loads trained model and feature engineer pipeline."""
        if not BEST_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model artifact not found at {BEST_MODEL_PATH}. Run training pipeline first!"
            )
        self.model = joblib.load(BEST_MODEL_PATH)
        self.feature_engineer = FeatureEngineer()

    def categorize_risk(self, prob: float) -> Tuple[str, str]:
        """Categorize numerical default probability into risk bands."""
        if prob < LOW_RISK_THRESHOLD:
            return "LOW RISK", "APPROVED"
        elif prob <= MEDIUM_RISK_THRESHOLD:
            return "MEDIUM RISK", "MANUAL REVIEW REQUIRED"
        else:
            return "HIGH RISK", "REJECTED"

    def predict_single(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs prediction for a single credit applicant input dictionary.
        Returns risk score, probability, risk level, recommendation, and top risk factors.
        """
        df_single = pd.DataFrame([input_data])
        X_processed = self.feature_engineer.transform(df_single)
        
        prob = float(self.model.predict_proba(X_processed)[0, 1])
        risk_level, recommendation = self.categorize_risk(prob)
        
        # Calculate individual feature risk contribution factors
        risk_factors = []
        if input_data.get("credit_score", 700) < 620:
            risk_factors.append("Low Credit Score (< 620)")
        if input_data.get("debt_to_income_ratio", 0.3) > 0.40:
            risk_factors.append("High Debt-to-Income Ratio (> 40%)")
        if input_data.get("credit_utilization_rate", 0.3) > 0.65:
            risk_factors.append("Elevated Credit Line Utilization (> 65%)")
        if input_data.get("payment_history_score", 80) < 65:
            risk_factors.append("Poor Historical Payment Track Record (< 65)")
        if input_data.get("loan_amount", 10000) / (input_data.get("annual_income", 50000) + 1e-5) > 0.45:
            risk_factors.append("Excessive Loan-to-Income Exposure (> 45%)")

        if not risk_factors:
            risk_factors.append("Healthy Overall Financial Profile")

        return {
            "default_probability": round(prob, 4),
            "risk_score_percentage": round(prob * 100, 1),
            "risk_level": risk_level,
            "recommendation": recommendation,
            "key_risk_factors": risk_factors
        }

    def predict_batch(self, df_batch: pd.DataFrame) -> List[Dict[str, Any]]:
        """Runs predictions for a batch DataFrame."""
        X_processed = self.feature_engineer.transform(df_batch)
        probs = self.model.predict_proba(X_processed)[:, 1]
        
        results = []
        for i, prob in enumerate(probs):
            prob_val = float(prob)
            risk_level, recommendation = self.categorize_risk(prob_val)
            results.append({
                "index": i,
                "default_probability": round(prob_val, 4),
                "risk_score_percentage": round(prob_val * 100, 1),
                "risk_level": risk_level,
                "recommendation": recommendation
            })
        return results
