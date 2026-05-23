from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from survey_contamination.detectors import (
    BusinessSuffixHeuristicDetector,
    FeatureClassifierDetector,
    NameDictionaryDetector,
    SpacyPersonDetector,
    build_name_sets,
)


OUTPUT_DIR = PROJECT_ROOT / "outputs"


def require_input(path: Path, producer_script: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run {producer_script} before this script.")


def main() -> None:
    input_path = OUTPUT_DIR / "synthetic_business_names.csv"
    surnames_path = OUTPUT_DIR / "surnames_top.csv"
    first_names_path = OUTPUT_DIR / "first_names_top.csv"
    require_input(input_path, "scripts/02_generate_contamination.py")
    require_input(surnames_path, "scripts/01_data_preparation.py")
    require_input(first_names_path, "scripts/01_data_preparation.py")

    frame = pd.read_csv(input_path)
    first_names = pd.read_csv(first_names_path).head(5_000)
    surnames = pd.read_csv(surnames_path).head(5_000)
    first_lookup, surname_lookup = build_name_sets(first_names, surnames)
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
