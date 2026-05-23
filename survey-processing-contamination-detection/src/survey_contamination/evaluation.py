from __future__ import annotations

import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


def evaluate_detector_predictions(frame: pd.DataFrame, detector_name: str) -> dict[str, float | int | str]:
    y_true = frame["is_contaminated"].astype(bool)
    y_pred = frame["predicted"].astype(bool)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[False, True]).ravel()
    return {
        "detector": detector_name,
        "records": int(len(frame)),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def summarize_error_examples(frame: pd.DataFrame, max_examples: int = 10) -> pd.DataFrame:
    false_positives = frame[(~frame["is_contaminated"].astype(bool)) & frame["predicted"].astype(bool)].copy()
    false_positives["error_type"] = "false_positive"
    false_negatives = frame[frame["is_contaminated"].astype(bool) & (~frame["predicted"].astype(bool))].copy()
    false_negatives["error_type"] = "false_negative"
    errors = pd.concat([false_positives, false_negatives], ignore_index=True)
    sort_columns = [column for column in ["error_type", "score"] if column in errors.columns]
    if sort_columns:
        errors = errors.sort_values(sort_columns, ascending=[True, False][: len(sort_columns)])
    return errors.head(max_examples).reset_index(drop=True)
