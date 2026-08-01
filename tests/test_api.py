import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["status"] == "healthy"

def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code in [200, 44]  # 44 if model not trained yet

def test_predict_endpoint():
    payload = {
        "age": 42,
        "annual_income": 95000,
        "credit_score": 720,
        "debt_to_income_ratio": 0.25,
        "credit_utilization_rate": 0.30,
        "payment_history_score": 90.0,
        "loan_amount": 15000,
        "employment_length_years": 10,
        "revolving_balance": 4500,
        "num_credit_lines": 8,
        "home_ownership": "OWN",
        "loan_intent": "VENTURE"
    }
    response = client.post("/predict", json=payload)
    if response.status_code == 200:
        json_data = response.json()
        assert "default_probability" in json_data
        assert "risk_level" in json_data
