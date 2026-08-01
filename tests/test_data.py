import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import pandas as pd
from src.data_loader import generate_credit_dataset
from src.feature_engineering import FeatureEngineer
from src.config import TARGET_COL

def test_dataset_generation():
    df = generate_credit_dataset(num_samples=200, random_state=42)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 200
    assert TARGET_COL in df.columns
    assert df[TARGET_COL].nunique() == 2

def test_feature_engineering_pipeline():
    df = generate_credit_dataset(num_samples=100, random_state=42)
    fe = FeatureEngineer()
    X_processed, y = fe.fit_transform(df)
    
    assert X_processed.shape[0] == 100
    assert "loan_to_income_ratio" in X_processed.columns
    assert "risk_score_index" in X_processed.columns
    assert not X_processed.isna().any().any()
