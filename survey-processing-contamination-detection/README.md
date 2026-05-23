# Survey Processing Contamination Detection

A practical lab for detecting personal names accidentally entered in business name fields using local rules, standard Python packages, and traditional machine learning.

## Why Not Start With an LLM?

This lab is intentionally not an LLM lab. The problem has structured clues, local reference data, measurable error trade-offs, and privacy considerations. Those properties make it a strong fit for deterministic rules and traditional ML before considering hosted AI services.

The goal is not to rule out LLMs forever. It is to practice choosing the smallest reliable tool for a targeted data quality problem, then measuring where that tool succeeds and fails.

## What You Will Build

You will build a runnable contamination detection pipeline that:

- prepares reference name data from local files
- creates synthetic business names with labeled contamination
- compares dictionary, heuristic, and feature-based ML detectors
- evaluates precision, recall, F1, true/false positive and negative counts, and error examples
- walks through the workflow in a teaching notebook

## Setup

Run all commands from this lab directory:

```bash
cd survey-processing-contamination-detection
```

```bash
conda create -n survey-contamination python=3.11
conda activate survey-contamination
pip install -r requirements.txt
```

If your system does not provide `python`, use `python3` for the commands below.

## Data Inputs

The lab uses bundled local inputs:

- `data/Names_2010Census.csv` for Census surnames
- `data/namesbystate.zip` for SSA first-name records by state

## Run the Pipeline

Run these commands from the `survey-processing-contamination-detection` lab directory:

```bash
python scripts/01_data_preparation.py
python scripts/02_generate_contamination.py
python scripts/03_run_detection.py
python scripts/04_evaluate.py
```

Each script bootstraps the local `src` package path, so you do not need to set `PYTHONPATH`.

## Run the Notebook

```bash
jupyter notebook notebooks/contamination_detection_lab.ipynb
```

The notebook follows the same lab flow as the scripts, with narrative explanations and tabular summaries for teaching or self-study.

## Optional spaCy Baseline

The core lab does not require spaCy. To try the optional NER detector:

```bash
pip install spacy
python -m spacy download en_core_web_lg
```

After installing spaCy and `en_core_web_lg`, rerun `scripts/03_run_detection.py` and `scripts/04_evaluate.py` to include `spacy_person`.

Use this as an additional baseline after you understand the local rules and traditional ML detectors. It is optional because the main learning objective is to evaluate targeted, locally runnable approaches first.

## Repository Structure

```text
survey-processing-contamination-detection/
|-- data/                         # Local reference datasets used by the lab
|-- notebooks/
|   `-- contamination_detection_lab.ipynb
|-- scripts/
|   |-- 01_data_preparation.py
|   |-- 02_generate_contamination.py
|   |-- 03_run_detection.py
|   `-- 04_evaluate.py
|-- src/
|   `-- survey_contamination/      # Data loading, synthetic data, detectors, evaluation
|-- tests/                         # Unit tests for lab components
|-- requirements.txt
`-- README.md
```

## Expected Outputs

Running the pipeline writes generated artifacts under `outputs/`. Treat these as reproducible run outputs rather than source files to maintain by hand.

- `outputs/surnames_top.csv`: prepared surname reference data; inspect the highest-frequency surnames used by the detectors and synthetic generator
- `outputs/first_names_top.csv`: prepared first-name reference data; inspect common first names available for contamination examples
- `outputs/synthetic_business_names.csv`: labeled synthetic dataset; inspect clean business names, contaminated rows, and the target labels
- `outputs/detector_scores.csv`: row-level detector outputs for the held-out test split; compare which records each detector flags
- `outputs/detector_metrics.csv`: precision, recall, F1, and true/false positive and negative counts; compare detector trade-offs
- `outputs/error_examples.csv`: false positives and false negatives; inspect concrete cases behind the metrics
- `outputs/detector_metrics.png`: metrics plot; use it as a quick visual comparison of detector performance

## Teaching Notes

This lab is useful for discussions about:

- when local reference data is enough
- how low contamination rates affect precision and recall
- why transparent rules can be valuable in review workflows
- how traditional ML can complement deterministic checks
- what evidence you would need before adding a hosted AI service
- why synthetic labels are useful for learning workflow mechanics, not proof of production performance on real survey data
