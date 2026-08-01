from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class CreditApplicantInput(BaseModel):
    age: int = Field(..., ge=18, le=100, example=35, description="Applicant age in years")
    annual_income: float = Field(..., ge=10000, le=1000000, example=75000.0, description="Annual gross income in USD")
    credit_score: int = Field(..., ge=300, le=850, example=710, description="FICO or equivalent credit score")
    debt_to_income_ratio: float = Field(..., ge=0.0, le=1.0, example=0.28, description="Debt-to-income ratio (0.0 to 1.0)")
    credit_utilization_rate: float = Field(..., ge=0.0, le=1.0, example=0.35, description="Revolving credit line utilization rate")
    payment_history_score: float = Field(..., ge=0.0, le=100.0, example=92.5, description="Historical payment score (0 to 100)")
    loan_amount: float = Field(..., ge=500, le=500000, example=15000.0, description="Requested loan amount in USD")
    employment_length_years: int = Field(..., ge=0, le=50, example=6, description="Years in current employment")
    revolving_balance: float = Field(..., ge=0.0, example=5400.0, description="Total current revolving credit balance")
    num_credit_lines: int = Field(..., ge=1, le=50, example=7, description="Number of open credit accounts")
    home_ownership: str = Field(..., example="MORTGAGE", description="Home ownership status: RENT, OWN, or MORTGAGE")
    loan_intent: str = Field(..., example="PERSONAL", description="Loan purpose: PERSONAL, EDUCATION, MEDICAL, VENTURE, or DEBTCONSOLIDATION")

class PredictionResponse(BaseModel):
    default_probability: float
    risk_score_percentage: float
    risk_level: str
    recommendation: str
    key_risk_factors: List[str]

class BatchPredictionResponse(BaseModel):
    total_processed: int
    low_risk_count: int
    medium_risk_count: int
    high_risk_count: int
    predictions: List[Dict[str, Any]]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    api_version: str

class MetricsResponse(BaseModel):
    best_model_name: str
    best_roc_auc: float
    models: Dict[str, Any]
    feature_importances: Dict[str, float]
