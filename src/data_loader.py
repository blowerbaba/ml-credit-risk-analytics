import numpy as np
import pandas as pd
from typing import Tuple
from src.config import RAW_DATA_PATH, TARGET_COL

def generate_credit_dataset(num_samples: int = 5000, random_state: int = 42) -> pd.DataFrame:
    """
    Generates a realistic financial credit risk dataset with synthetic attributes,
    non-linear feature relationships, and ~18-22% imbalanced default rate.
    """
    np.random.seed(random_state)
    
    # 1. Primary Demographics & Financials
    age = np.random.randint(21, 68, size=num_samples)
    annual_income = np.random.lognormal(mean=11.0, sigma=0.6, size=num_samples).round(-2)
    annual_income = np.clip(annual_income, 18000, 350000)
    
    credit_score = np.random.normal(loc=670, scale=75, size=num_samples).astype(int)
    credit_score = np.clip(credit_score, 350, 850)
    
    debt_to_income_ratio = np.random.beta(a=2, b=5, size=num_samples) * 0.70
    debt_to_income_ratio = np.round(debt_to_income_ratio, 3)
    
    credit_utilization_rate = np.random.beta(a=2.5, b=3.5, size=num_samples)
    credit_utilization_rate = np.round(credit_utilization_rate, 3)
    
    payment_history_score = np.random.normal(loc=82, scale=12, size=num_samples)
    payment_history_score = np.clip(payment_history_score, 30, 100).round(1)
    
    loan_amount = np.random.lognormal(mean=9.5, sigma=0.7, size=num_samples).round(-2)
    loan_amount = np.clip(loan_amount, 2000, 75000)
    
    employment_length_years = np.random.exponential(scale=5, size=num_samples).astype(int)
    employment_length_years = np.clip(employment_length_years, 0, 35)
    
    revolving_balance = (annual_income * np.random.uniform(0.05, 0.40, size=num_samples) * credit_utilization_rate).round(0)
    num_credit_lines = np.random.poisson(lam=6, size=num_samples)
    num_credit_lines = np.clip(num_credit_lines, 1, 25)
    
    # Categorical Attributes
    home_ownership = np.random.choice(
        ["RENT", "MORTGAGE", "OWN"], size=num_samples, p=[0.45, 0.43, 0.12]
    )
    loan_intent = np.random.choice(
        ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "DEBTCONSOLIDATION"],
        size=num_samples,
        p=[0.25, 0.20, 0.20, 0.15, 0.20]
    )
    
    # Vectorized Log-Odds calculation
    log_odds = (
        - 2.5
        + 0.025 * (700 - credit_score) / 100
        + 3.5 * debt_to_income_ratio
        + 2.8 * credit_utilization_rate
        - 0.04 * (payment_history_score - 70)
        + 1.8 * (loan_amount / annual_income)
        - 0.05 * employment_length_years
        + np.where(home_ownership == "RENT", 1.2, 0.0)
        + np.where(loan_intent == "MEDICAL", 0.8, 0.0)
        + np.random.normal(0, 0.6, size=num_samples)  # Random noise
    )
    
    # Convert log-odds to probability via sigmoid
    prob_default = 1 / (1 + np.exp(-log_odds))
    
    # Target variable (1 = Default Risk, 0 = Safe Borrower)
    default_risk = (prob_default > 0.62).astype(int)
    
    df = pd.DataFrame({
        "age": age,
        "annual_income": annual_income,
        "credit_score": credit_score,
        "debt_to_income_ratio": debt_to_income_ratio,
        "credit_utilization_rate": credit_utilization_rate,
        "payment_history_score": payment_history_score,
        "loan_amount": loan_amount,
        "employment_length_years": employment_length_years,
        "revolving_balance": revolving_balance,
        "num_credit_lines": num_credit_lines,
        "home_ownership": home_ownership,
        "loan_intent": loan_intent,
        TARGET_COL: default_risk
    })
    
    return df

def load_or_create_data(save_path=RAW_DATA_PATH) -> pd.DataFrame:
    """Loads existing raw dataset or creates a fresh dataset if missing."""
    if save_path.exists():
        df = pd.read_csv(save_path)
    else:
        df = generate_credit_dataset(num_samples=5000)
        df.to_csv(save_path, index=False)
    return df

if __name__ == "__main__":
    df = load_or_create_data()
    print(f"Dataset Loaded Successfully! Shape: {df.shape}")
    print(f"Default Risk Distribution:\n{df[TARGET_COL].value_counts(normalize=True)}")
