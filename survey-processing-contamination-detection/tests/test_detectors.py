import pandas as pd

from survey_contamination.detectors import (
    BusinessSuffixHeuristicDetector,
    FeatureClassifierDetector,
    NameDictionaryDetector,
)


def test_name_dictionary_detector_flags_person_name():
    detector = NameDictionaryDetector(first_names={"John", "Maria"}, surnames={"Smith", "Garcia"})

    predictions = detector.predict(["John Smith", "Northstar Analytics LLC"])

    assert predictions.tolist() == [True, False]


def test_business_suffix_heuristic_avoids_business_suffix_false_positive():
    detector = BusinessSuffixHeuristicDetector(first_names={"John"}, surnames={"Smith"})

    predictions = detector.predict(["John Smith", "Smith Consulting LLC"])

    assert predictions.tolist() == [True, False]


def test_business_suffix_heuristic_accepts_generator_input():
    detector = BusinessSuffixHeuristicDetector(first_names={"John"}, surnames={"Smith"})
    values = (value for value in ["John Smith", "Smith Consulting LLC"])

    predictions = detector.predict(values)

    assert predictions.tolist() == [True, False]


def test_feature_classifier_detector_learns_obvious_examples():
    training = pd.DataFrame(
        {
            "business_name": ["John Smith", "Maria Garcia", "Northstar Logistics LLC", "Summit Dental Inc"],
            "is_contaminated": [True, True, False, False],
        }
    )
    detector = FeatureClassifierDetector(first_names={"John", "Maria"}, surnames={"Smith", "Garcia"})
    detector.fit(training["business_name"], training["is_contaminated"])

    scores = detector.score(["John Garcia", "Riverbend Services LLC"])
    predictions = detector.predict(["John Garcia", "Riverbend Services LLC"])

    assert scores.shape == (2,)
    assert predictions.tolist() == [True, False]


def test_feature_classifier_detector_scores_only_clean_training_as_zero():
    detector = FeatureClassifierDetector(first_names={"John", "Maria"}, surnames={"Smith", "Garcia"})
    detector.fit(["Northstar Logistics LLC", "Summit Dental Inc"], [False, False])

    scores = detector.score(["John Smith", "Riverbend Services LLC"])
    predictions = detector.predict(["John Smith", "Riverbend Services LLC"])

    assert scores.tolist() == [0.0, 0.0]
    assert predictions.tolist() == [False, False]


def test_feature_classifier_detector_scores_only_contaminated_training_as_one():
    detector = FeatureClassifierDetector(first_names={"John", "Maria"}, surnames={"Smith", "Garcia"})
    detector.fit(["John Smith", "Maria Garcia"], [True, True])

    scores = detector.score(["John Smith", "Riverbend Services LLC"])
    predictions = detector.predict(["John Smith", "Riverbend Services LLC"])

    assert scores.tolist() == [1.0, 1.0]
    assert predictions.tolist() == [True, True]
