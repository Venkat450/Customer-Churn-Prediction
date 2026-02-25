"""Evaluation helpers: cross-validation, metrics, plots, and coefficient inspection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import StratifiedKFold

def cross_validate_metrics(model, X, y, n_splits: int = 5, random_state: int = 42) -> Dict[str, float]:
    """Run stratified K-fold CV and report mean metrics."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    aucs, precs, recs, f1s = [], [], [], []

    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_va)[:, 1]
        pred = (proba >= 0.5).astype(int)

        aucs.append(roc_auc_score(y_va, proba))
        precs.append(precision_score(y_va, pred, zero_division=0))
        recs.append(recall_score(y_va, pred, zero_division=0))
        f1s.append(f1_score(y_va, pred, zero_division=0))

    return {
        "cv_roc_auc_mean": float(np.mean(aucs)),
        "cv_precision_mean": float(np.mean(precs)),
        "cv_recall_mean": float(np.mean(recs)),
        "cv_f1_mean": float(np.mean(f1s)),
        "cv_splits": int(n_splits),
    }

def test_metrics(model, X_test, y_test, threshold: float = 0.5) -> Dict[str, float]:
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)

    return {
        "test_roc_auc": float(roc_auc_score(y_test, proba)),
        "test_precision": float(precision_score(y_test, pred, zero_division=0)),
        "test_recall": float(recall_score(y_test, pred, zero_division=0)),
        "test_f1": float(f1_score(y_test, pred, zero_division=0)),
        "test_accuracy": float(accuracy_score(y_test, pred)),
        "threshold": float(threshold),
    }

def save_json(d: Dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(d, indent=2))

def make_plots(model, X_test, y_test, out_roc: str, out_pr: str, out_cm: str, threshold: float = 0.5) -> None:
    """Save ROC, PR curves and confusion matrix."""
    Path(out_roc).parent.mkdir(parents=True, exist_ok=True)

    # ROC
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title("ROC Curve (Test Set)")
    plt.savefig(out_roc, bbox_inches="tight", dpi=200)
    plt.close()

    # PR
    PrecisionRecallDisplay.from_estimator(model, X_test, y_test)
    plt.title("Precision-Recall Curve (Test Set)")
    plt.savefig(out_pr, bbox_inches="tight", dpi=200)
    plt.close()

    # Confusion matrix at threshold
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    cm = confusion_matrix(y_test, pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["No Churn", "Churn"])
    disp.plot()
    plt.title(f"Confusion Matrix (threshold={threshold})")
    plt.savefig(out_cm, bbox_inches="tight", dpi=200)
    plt.close()

def top_coefficients(fitted_pipeline, top_k: int = 20) -> pd.DataFrame:
    """Extract top +/- coefficients from a fitted logistic regression pipeline."""
    pre = fitted_pipeline.named_steps["preprocess"]
    model = fitted_pipeline.named_steps["model"]

    feature_names = pre.get_feature_names_out()
    coefs = model.coef_.ravel()

    df = pd.DataFrame({"feature": feature_names, "coef": coefs})
    df["abs_coef"] = df["coef"].abs()
    df = df.sort_values("abs_coef", ascending=False)

    top = df.head(top_k).copy()
    top["direction"] = np.where(top["coef"] >= 0, "increases churn", "decreases churn")
    return top[["feature", "coef", "direction"]]
