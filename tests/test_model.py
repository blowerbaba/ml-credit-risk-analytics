import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import pandas as pd
from src.predictor import CreditRiskPredictor

def test_predictor_single_sample():
    predictor = CreditRiskPredictor()
    sample = {
        "age": 30,
        "annual_income": 80000,
        "credit_score": 750,
        "debt_to_income_ratio": 0.20,
        "credit_utilization_rate": 0.25,
        "payment_history_score": 95.0,
        "loan_amount": 10000,
        "employment_length_years": 8,
        "revolving_balance": 3000,
        "num_credit_lines": 5,
        "home_ownership": "MORTGAGE",
        "loan_intent": "PERSONAL"
    }
    
    output = predictor.predict_single(sample)
    
    assert "default_probability" in output
    assert 0.0 <= output["default_probability"] <= 1.0
    assert output["risk_level"] in ["LOW RISK", "MEDIUM RISK", "HIGH RISK"]
    assert isinstance(output["key_risk_factors"], list)
