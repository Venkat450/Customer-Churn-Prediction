"""Train + evaluate the churn model end-to-end.
"""

from __future__ import annotations

import argparse
from datetime import datetime

import pandas as pd
from sklearn.model_selection import train_test_split
import joblib

from .data import download_data
from .preprocess import clean_dataframe, split_xy, build_preprocessor
from .model import build_model
from .evaluate import cross_validate_metrics, test_metrics, make_plots, save_json, top_coefficients
from .threshold import find_best_threshold
from .config import (
    RAW_DATA_PATH, ARTIFACT_DIR, MODEL_PATH, METRICS_PATH, ROC_PATH, PR_PATH, CM_PATH, COEF_PATH
)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--force-download", action="store_true", help="Re-download the dataset.")
    p.add_argument("--test-size", type=float, default=0.2, help="Test set proportion.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for churn class.")
    p.add_argument("--cv-splits", type=int, default=5, help="Number of CV folds.")
    p.add_argument("--optimize", type=str, default=None, help="Threshold optimization: f1 | recall | cost")
    return p.parse_args()

def main():
    args = parse_args()

    # 1) Acquire data
    path = download_data(out_path=RAW_DATA_PATH, force=args.force_download)

    # 2) Load + clean
    df = pd.read_csv(path)
    df = clean_dataframe(df)

    # Drop rows with missing target or missing key numeric after coercion (rare)
    df = df.dropna(subset=["Churn"]).reset_index(drop=True)

    X, y = split_xy(df)

    # 3) Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    # 4) Build pipeline
    pre = build_preprocessor(X_train)
    pipe = build_model(pre)

    # 5) Cross-validation on train
    cv = cross_validate_metrics(pipe, X_train, y_train, n_splits=args.cv_splits, random_state=args.seed)

    # 6) Fit full model on train
    pipe.fit(X_train, y_train)

    
    # 7) Optional threshold optimization
    if args.optimize:
        best_t, score = find_best_threshold(pipe, X_train, y_train, strategy=args.optimize)
        print(f"Optimized threshold ({args.optimize}):", best_t)
        args.threshold = best_t

    # 8) Test evaluation + plots

    tst = test_metrics(pipe, X_test, y_test, threshold=args.threshold)
    make_plots(pipe, X_test, y_test, ROC_PATH, PR_PATH, CM_PATH, threshold=args.threshold)

    # 8) Save artifacts
    joblib.dump(pipe, MODEL_PATH)
    top = top_coefficients(pipe, top_k=30)
    top.to_csv(COEF_PATH, index=False)

    metrics = {
        "run_timestamp": datetime.utcnow().isoformat() + "Z",
        "data_path": path,
        **cv,
        **tst,
        "n_rows": int(df.shape[0]),
        "n_features_raw": int(X.shape[1]),
    }
    save_json(metrics, METRICS_PATH)

    print("Training complete.")
    print("Saved:", MODEL_PATH)
    print("Saved:", METRICS_PATH)
    print("Saved:", ROC_PATH)
    print("Saved:", PR_PATH)
    print("Saved:", CM_PATH)
    print("Saved:", COEF_PATH)

if __name__ == "__main__":
    main()
