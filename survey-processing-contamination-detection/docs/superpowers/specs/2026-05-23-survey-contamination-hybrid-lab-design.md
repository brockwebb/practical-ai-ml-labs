# Survey Contamination Detection Hybrid Lab Design

## Purpose

Build the current README concept into a runnable hybrid lab for detecting personal names accidentally entered in business-name survey fields. The lab should feel practical for survey processing and data quality work, while staying light enough for students to run locally without specialized hardware.

## Scope

The first implementation will create a complete baseline lab, not a research-grade benchmark. It will include reusable Python modules, command-line scripts, a notebook walkthrough, tests, and updated setup instructions.

In scope:

- Load local Census surname data from `data/Names_2010Census.csv`.
- Load SSA first-name data directly from `data/namesbystate.zip`.
- Generate synthetic but realistic business names.
- Inject personal-name contamination at configurable low rates.
- Run rule-based, heuristic, and lightweight ML detectors.
- Optionally run a spaCy NER baseline when spaCy and an English model are installed.
- Evaluate precision, recall, F1, confusion matrices, and false-positive examples.
- Produce tabular outputs and plots suitable for discussion in a lab setting.

Out of scope for the first pass:

- Transformer-based models.
- Production deployment services or APIs.
- Download automation for external datasets.
- Manual labeling interfaces.

## Recommended Approach

Use a hybrid structure:

- A notebook is the primary teaching path.
- Scripts provide a reproducible pipeline that matches the README.
- Shared implementation lives under `src/` so notebook and scripts use the same logic.

This gives students readable explanations and repeatable engineering structure without duplicating logic.

## Repository Structure

```text
notebooks/
  contamination_detection_lab.ipynb
scripts/
  01_data_preparation.py
  02_generate_contamination.py
  03_run_detection.py
  04_evaluate.py
src/
  survey_contamination/
    __init__.py
    data.py
    synthetic.py
    detectors.py
    features.py
    evaluation.py
tests/
  test_data.py
  test_synthetic.py
  test_detectors.py
requirements.txt
README.md
```

## Data Flow

1. `data.py` loads surnames and first names from local files.
2. `synthetic.py` creates clean business names using industry terms, owner-name patterns, suffixes, and geography-like tokens.
3. The same module injects contaminated rows such as `John Smith`, `Maria Garcia`, and `A. Johnson`.
4. `detectors.py` applies multiple detection strategies to each business-name value.
5. `evaluation.py` compares detector predictions with known synthetic labels.
6. Scripts write intermediate CSVs and summary outputs under an ignored `outputs/` directory.

## Detection Methods

The lab will include these detectors:

- `NameDictionaryDetector`: flags rows where tokens strongly match common first and last names.
- `BusinessSuffixHeuristicDetector`: reduces false positives when names include strong business signals such as `LLC`, `Inc`, `Company`, `Services`, or `Associates`.
- `FeatureClassifierDetector`: trains a scikit-learn classifier using token, character, suffix, and dictionary-match features.
- `SpacyPersonDetector`: optional baseline that uses spaCy NER when installed; skipped with a clear message otherwise.

The detector interface should be simple: each detector accepts a sequence of strings and returns scores or binary predictions.

## Notebook Flow

The notebook will walk through:

1. Problem framing and why low contamination rates are hard.
2. Loading names and inspecting frequency distributions.
3. Creating synthetic businesses and contaminated examples.
4. Running each detector.
5. Comparing metrics across contamination rates.
6. Reviewing false positives and false negatives.
7. Discussing threshold selection for survey operations.

The notebook should be executable top to bottom after installing `requirements.txt`.

## Error Handling

- Missing required data files should raise clear `FileNotFoundError` messages with expected paths.
- Optional spaCy dependencies should not break the lab; the spaCy detector should be skipped if unavailable.
- Scripts should create output directories as needed.
- Randomness should be controlled with a configurable seed.

## Testing

Add focused tests for:

- Loading surname and SSA first-name files.
- Generating clean and contaminated records with expected labels.
- Detector behavior on obvious clean and contaminated examples.
- Metric calculation shape and expected fields.

The verification target is:

```bash
python -m pytest
python scripts/01_data_preparation.py
python scripts/02_generate_contamination.py
python scripts/03_run_detection.py
python scripts/04_evaluate.py
```

## Success Criteria

The lab is complete when:

- A fresh user can follow the README to install dependencies and run the notebook or scripts.
- The scripts produce reusable intermediate datasets and evaluation summaries.
- Tests pass without requiring spaCy.
- spaCy is documented as optional.
- The notebook demonstrates the practical precision-recall trade-off at low contamination rates.
