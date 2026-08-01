import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.train import train_and_benchmark_models
from src.predictor import CreditRiskPredictor

def run_end_to_end_pipeline():
    """CLI orchestrator for the entire ML pipeline."""
    print("=" * 70)
    print("STARTING END-TO-END ML ENGINEER PIPELINE EXECUTION")
    print("=" * 70)
    start_time = time.time()
    
    # 1. Train and Benchmark Models
    results = train_and_benchmark_models()
    
    print("\n" + "=" * 70)
    print("VERIFYING INFERENCE ENGINE WITH SAMPLE PREDICTION...")
    print("=" * 70)
    
    # 2. Test Predictor
    predictor = CreditRiskPredictor()
    sample_applicant = {
        "age": 34,
        "annual_income": 65000,
        "credit_score": 580,
        "debt_to_income_ratio": 0.48,
        "credit_utilization_rate": 0.72,
        "payment_history_score": 60.0,
        "loan_amount": 25000,
        "employment_length_years": 3,
        "revolving_balance": 18000,
        "num_credit_lines": 8,
        "home_ownership": "RENT",
        "loan_intent": "DEBTCONSOLIDATION"
    }
    
    prediction = predictor.predict_single(sample_applicant)
    print(f"Sample Applicant Test Output:")
    print(f"  - Risk Score: {prediction['risk_score_percentage']}%")
    print(f"  - Risk Level: {prediction['risk_level']}")
    print(f"  - Recommendation: {prediction['recommendation']}")
    print(f"  - Key Risk Factors: {', '.join(prediction['key_risk_factors'])}")
    
    elapsed = time.time() - start_time
    print("=" * 70)
    print(f"PIPELINE COMPLETED SUCCESSFULLY IN {elapsed:.2f} SECONDS!")
    print("=" * 70)

if __name__ == "__main__":
    run_end_to_end_pipeline()
