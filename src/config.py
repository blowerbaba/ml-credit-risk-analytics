import os
from pathlib import Path

# Base Directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
WEB_DIR = BASE_DIR / "web"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Data Files
RAW_DATA_PATH = DATA_DIR / "credit_risk_dataset.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed_credit_data.csv"

# Model Artifact Paths
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
SCALER_PATH = MODELS_DIR / "scaler.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"
SHAP_SUMMARY_PATH = MODELS_DIR / "shap_summary.json"

# Features Configuration
TARGET_COL = "default_risk"

NUMERICAL_FEATURES = [
    "age",
    "annual_income",
    "credit_score",
    "debt_to_income_ratio",
    "credit_utilization_rate",
    "payment_history_score",
    "loan_amount",
    "employment_length_years",
    "revolving_balance",
    "num_credit_lines",
]

CATEGORICAL_FEATURES = [
    "home_ownership",   # RENT, OWN, MORTGAGE
    "loan_intent",      # PERSONAL, EDUCATION, MEDICAL, VENTURE, DEBTCONSOLIDATION
]

FEATURE_COLUMNS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# Engineered Feature Names (created dynamically during preprocessing)
ENGINEERED_NUMERICAL_FEATURES = [
    "loan_to_income_ratio",
    "revolving_utilization_index",
    "risk_score_index"
]

ALL_NUMERICAL_FEATURES = NUMERICAL_FEATURES + ENGINEERED_NUMERICAL_FEATURES

# Risk Categorization Thresholds
LOW_RISK_THRESHOLD = 0.35      # < 35% probability of default -> LOW RISK (Approved)
MEDIUM_RISK_THRESHOLD = 0.65   # 35% - 65% probability -> MEDIUM RISK (Manual Review)
                               # > 65% probability -> HIGH RISK (Rejected)
