"""Model definition."""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

def build_model(preprocessor) -> Pipeline:
    """Build a preprocessing + logistic regression pipeline.

    Notes:
    - class_weight='balanced' helps when the churn class is smaller.
    - solver='lbfgs' works well with one-hot features.
    """
    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        n_jobs=None,
        solver="lbfgs",
    )
    pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", clf)])
    return pipe
