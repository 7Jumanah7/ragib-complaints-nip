"""Sliced evaluation, bootstrap confidence intervals and error taxonomy (Lab 6)."""
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


def sliced_accuracy(records, columns=("language", "dialect", "length")):
    """Accuracy and sample count for each slice. `records` need a 'correct' field (0/1)."""
    df = pd.DataFrame(records)
    return {col: df.groupby(col)["correct"].agg(accuracy="mean", n="count").round(2)
            for col in columns}


def bootstrap_ci(y_true, y_pred, n_resamples=2000, seed=7, alpha=5):
    """Bootstrap (100 - alpha)% confidence interval for accuracy."""
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    scores = []
    for _ in range(n_resamples):
        i = rng.choice(len(y_true), len(y_true), replace=True)
        scores.append(accuracy_score(y_true[i], y_pred[i]))
    return (float(np.percentile(scores, alpha / 2)),
            float(np.percentile(scores, 100 - alpha / 2)))


def error_taxonomy(errors):
    """Count error types. `errors` = (text, actual, predicted, error_type) tuples."""
    return Counter(e[3] for e in errors if e[1] != e[2])
