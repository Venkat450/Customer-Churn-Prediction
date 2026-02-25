"""Automatic threshold selection utilities.

Supports:
- Maximize Recall (subject to minimum precision)
- Maximize F1
- Cost-based optimization
"""

import numpy as np
from sklearn.metrics import precision_recall_curve

def find_best_threshold(model, X, y, strategy="f1",
                        min_precision=0.0,
                        cost_fp=1.0,
                        cost_fn=5.0):
    """
    strategy:
        - 'f1'
        - 'recall'
        - 'cost'
    """
    probs = model.predict_proba(X)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y, probs)

    thresholds = np.append(thresholds, 1.0)

    best_threshold = 0.5
    best_score = -np.inf

    for p, r, t in zip(precision, recall, thresholds):
        if strategy == "f1":
            if (p + r) == 0:
                continue
            score = 2 * (p * r) / (p + r)
        elif strategy == "recall":
            if p < min_precision:
                continue
            score = r
        elif strategy == "cost":
            # Expected cost approximation
            preds = (probs >= t).astype(int)
            fp = ((preds == 1) & (y == 0)).sum()
            fn = ((preds == 0) & (y == 1)).sum()
            score = -(cost_fp * fp + cost_fn * fn)
        else:
            raise ValueError("Invalid strategy")

        if score > best_score:
            best_score = score
            best_threshold = t

    return float(best_threshold), float(best_score)
