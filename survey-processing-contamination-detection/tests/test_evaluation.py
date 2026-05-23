import pandas as pd

from survey_contamination.evaluation import evaluate_detector_predictions, summarize_error_examples


def test_evaluate_detector_predictions_returns_metrics():
    frame = pd.DataFrame(
        {
            "is_contaminated": [True, True, False, False],
            "predicted": [True, False, False, True],
            "score": [0.9, 0.4, 0.1, 0.8],
        }
    )

    metrics = evaluate_detector_predictions(frame, detector_name="demo")

    assert metrics["detector"] == "demo"
    assert metrics["true_positives"] == 1
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 1
    assert 0 <= metrics["f1"] <= 1


def test_summarize_error_examples_returns_false_positive_and_negative_rows():
    frame = pd.DataFrame(
        {
            "business_name": ["John Smith", "Northstar LLC"],
            "is_contaminated": [True, False],
            "predicted": [False, True],
            "score": [0.2, 0.9],
        }
    )

    errors = summarize_error_examples(frame, max_examples=5)

    assert set(errors["error_type"]) == {"false_negative", "false_positive"}
