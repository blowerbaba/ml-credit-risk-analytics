import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional
from src.config import (
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    ENGINEERED_NUMERICAL_FEATURES,
    ALL_NUMERICAL_FEATURES,
    SCALER_PATH,
    TARGET_COL
)

class FeatureEngineer:
    """
    Feature engineering pipeline for Financial Credit Risk assessment.
    Handles domain feature construction, categorical one-hot encoding,
    and standard feature scaling.
    """
    def __init__(self):
        self.scaler = StandardScaler()
        self.fitted = False
        self.encoded_categorical_cols = []

    def create_engineered_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Construct domain-specific financial ratios and indexes."""
        data = df.copy()
        
        # 1. Loan to Income Ratio
        data["loan_to_income_ratio"] = np.round(
            data["loan_amount"] / (data["annual_income"] + 1e-5), 4
        )
        
        # 2. Revolving Utilization Index
        data["revolving_utilization_index"] = np.round(
            data["revolving_balance"] / (data["annual_income"] * 0.3 + 1e-5), 4
        )
        
        # 3. Risk Score Index (Combined score & debt pressure)
        data["risk_score_index"] = np.round(
            (850 - data["credit_score"]) * data["debt_to_income_ratio"], 4
        )
        
        return data

    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Fit scaler on training data and transform features."""
        data = self.create_engineered_features(df)
        
        # Separate target if present
        y = data[TARGET_COL] if TARGET_COL in data.columns else None
        
        # Select numeric features
        X_num = data[ALL_NUMERICAL_FEATURES].copy()
        
        # Fit & Transform Scaler on Numerical Features
        X_num_scaled = self.scaler.fit_transform(X_num)
        X_num_df = pd.DataFrame(X_num_scaled, columns=ALL_NUMERICAL_FEATURES, index=df.index)
        
        # One-Hot Encode Categorical Features
        X_cat = pd.get_dummies(data[CATEGORICAL_FEATURES], drop_first=False)
        self.encoded_categorical_cols = list(X_cat.columns)
        
        # Concatenate numerical and categorical
        X_processed = pd.concat([X_num_df, X_cat], axis=1)
        
        self.fitted = True
        
        # Save pipeline artifact bundle
        artifact_bundle = {
            "scaler": self.scaler,
            "encoded_categorical_cols": self.encoded_categorical_cols
        }
        joblib.dump(artifact_bundle, SCALER_PATH)
        
        return X_processed, y

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted scaler and consistent column encoding."""
        if not self.fitted:
            if SCALER_PATH.exists():
                artifact_bundle = joblib.load(SCALER_PATH)
                if isinstance(artifact_bundle, dict):
                    self.scaler = artifact_bundle["scaler"]
                    self.encoded_categorical_cols = artifact_bundle["encoded_categorical_cols"]
                else:
                    self.scaler = artifact_bundle
                self.fitted = True
            else:
                raise ValueError("FeatureEngineer must be fitted before calling transform!")
                
        data = self.create_engineered_features(df)
        
        X_num = data[ALL_NUMERICAL_FEATURES].copy()
        X_num_scaled = self.scaler.transform(X_num)
        X_num_df = pd.DataFrame(X_num_scaled, columns=ALL_NUMERICAL_FEATURES, index=df.index)
        
        # Get one-hot dummies and align columns
        X_cat = pd.get_dummies(data[CATEGORICAL_FEATURES], drop_first=False)
        
        # Always align dummy columns to match fit_transform
        if self.encoded_categorical_cols:
            X_cat = X_cat.reindex(columns=self.encoded_categorical_cols, fill_value=0)
            
        X_processed = pd.concat([X_num_df, X_cat], axis=1)
        return X_processed

if __name__ == "__main__":
    from src.data_loader import load_or_create_data
    df = load_or_create_data()
    fe = FeatureEngineer()
    X, y = fe.fit_transform(df)
    print(f"Feature Engineering Complete! Processed shape: {X.shape}")
