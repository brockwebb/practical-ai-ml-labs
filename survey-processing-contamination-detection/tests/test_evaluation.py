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
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5


def test_evaluate_detector_predictions_handles_empty_frame():
    frame = pd.DataFrame(
        {
            "is_contaminated": pd.Series(dtype=bool),
            "predicted": pd.Series(dtype=bool),
            "score": pd.Series(dtype=float),
        }
    )

    metrics = evaluate_detector_predictions(frame, detector_name="empty")

    assert metrics == {
        "detector": "empty",
        "records": 0,
        "true_positives": 0,
        "false_positives": 0,
        "true_negatives": 0,
        "false_negatives": 0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
    }


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


def test_summarize_error_examples_returns_empty_frame_when_no_errors():
    frame = pd.DataFrame(
        {
            "business_name": ["John Smith", "Northstar LLC"],
            "is_contaminated": [True, False],
            "predicted": [True, False],
            "score": [0.9, 0.1],
        }
    )

    errors = summarize_error_examples(frame)

    assert errors.empty
    assert "error_type" in errors.columns


def test_summarize_error_examples_works_without_score_column():
    frame = pd.DataFrame(
        {
            "business_name": ["John Smith", "Northstar LLC"],
            "is_contaminated": [True, False],
            "predicted": [False, True],
        }
    )

    errors = summarize_error_examples(frame)

    assert list(errors["error_type"]) == ["false_negative", "false_positive"]


def test_summarize_error_examples_caps_and_sorts_by_error_type_then_score():
    frame = pd.DataFrame(
        {
            "business_name": [
                "John Smith",
                "Jane Brown",
                "Northstar LLC",
                "Acme Inc",
                "Taylor Jones",
            ],
            "is_contaminated": [True, True, False, False, True],
            "predicted": [False, False, True, True, False],
            "score": [0.2, 0.8, 0.9, 0.3, 0.6],
        }
    )

    errors = summarize_error_examples(frame, max_examples=4)

    assert len(errors) == 4
    assert list(errors["error_type"]) == [
        "false_negative",
        "false_negative",
        "false_negative",
        "false_positive",
    ]
    assert list(errors["score"]) == [0.8, 0.6, 0.2, 0.9]
