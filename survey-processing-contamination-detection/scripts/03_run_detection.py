from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from survey_contamination.data import load_first_names, load_surnames
from survey_contamination.detectors import (
    BusinessSuffixHeuristicDetector,
    FeatureClassifierDetector,
    NameDictionaryDetector,
    SpacyPersonDetector,
    build_name_sets,
)


OUTPUT_DIR = Path("outputs")


def main() -> None:
    input_path = OUTPUT_DIR / "synthetic_business_names.csv"
    if not input_path.exists():
        raise FileNotFoundError("Run scripts/02_generate_contamination.py before this script.")

    frame = pd.read_csv(input_path)
    first_lookup, surname_lookup = build_name_sets(load_first_names(top_n=5_000), load_surnames(top_n=5_000))
    train, test = train_test_split(
        frame,
        test_size=0.35,
        random_state=42,
        stratify=frame["is_contaminated"],
    )

    detectors = [
        NameDictionaryDetector(first_lookup, surname_lookup),
        BusinessSuffixHeuristicDetector(first_lookup, surname_lookup),
    ]

    classifier = FeatureClassifierDetector(first_lookup, surname_lookup)
    classifier.fit(train["business_name"], train["is_contaminated"])
    detectors.append(classifier)

    try:
        detectors.append(SpacyPersonDetector())
    except RuntimeError as exc:
        print(f"Skipping optional spaCy detector: {exc}")

    scored_frames: list[pd.DataFrame] = []
    for detector in detectors:
        scored = test.copy()
        scored["detector"] = detector.name
        scored["score"] = detector.score(scored["business_name"])
        scored["predicted"] = detector.predict(scored["business_name"])
        scored_frames.append(scored)

    output = pd.concat(scored_frames, ignore_index=True)
    output_path = OUTPUT_DIR / "detector_scores.csv"
    output.to_csv(output_path, index=False)
    print(f"Wrote detector scores to {output_path}")


if __name__ == "__main__":
    main()
