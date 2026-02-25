"""Preprocessing and feature engineering.

Key steps:
- Clean known data quality issues (e.g., TotalCharges blank strings)
- Separate target label
- Build a scikit-learn ColumnTransformer for numeric & categorical features
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

TARGET_COL = "Churn"
ID_COL = "customerID"

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw Telco churn dataset.

    - Convert TotalCharges to numeric (coerce blanks to NaN)
    - Standardize churn label
    """
    df = df.copy()

    # TotalCharges sometimes includes blank strings (customers with very short tenure).
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Ensure target is consistent
    if TARGET_COL in df.columns:
        df[TARGET_COL] = df[TARGET_COL].map({"Yes": 1, "No": 0}).astype("Int64")

    return df

def split_xy(df: pd.DataFrame):
    """Split dataframe into X and y, dropping ID column."""
    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COL}")

    y = df[TARGET_COL].astype(int)
    X = df.drop(columns=[TARGET_COL], errors="ignore")
    X = X.drop(columns=[ID_COL], errors="ignore")
    return X, y

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Create a ColumnTransformer for numeric + categorical features."""
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return pre
