# Survey Contamination Hybrid Lab Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a runnable hybrid lab that teaches local rules and traditional ML for detecting personal-name contamination in business-name survey fields.

**Architecture:** Shared Python modules under `src/survey_contamination/` own the reusable logic. Four scripts provide a reproducible command-line pipeline, and one notebook provides the teaching walkthrough using the same module functions. Outputs are written under `outputs/`, which is generated at runtime and not required in git.

**Tech Stack:** Python 3.11, pandas, numpy, scikit-learn, matplotlib, seaborn, pytest, optional spaCy.

---

## File Structure

- Create `requirements.txt`: minimal runtime and test dependencies, with spaCy documented as optional in README rather than required.
- Create `src/survey_contamination/__init__.py`: package exports and version string.
- Create `src/survey_contamination/data.py`: load Census surnames and SSA first names from local files.
- Create `src/survey_contamination/synthetic.py`: generate clean business names and injected personal-name contamination.
- Create `src/survey_contamination/features.py`: convert names into lightweight feature frames for traditional ML.
- Create `src/survey_contamination/detectors.py`: implement dictionary, heuristic, classifier, and optional spaCy detectors behind a common interface.
- Create `src/survey_contamination/evaluation.py`: metrics, confusion matrices, detector comparison tables, and error samples.
- Create `scripts/01_data_preparation.py`: validate and summarize source name files.
- Create `scripts/02_generate_contamination.py`: write synthetic labeled examples.
- Create `scripts/03_run_detection.py`: run all available detectors and write scored output.
- Create `scripts/04_evaluate.py`: write metrics and plots.
- Create `notebooks/contamination_detection_lab.ipynb`: teaching walkthrough.
- Create `tests/test_data.py`: unit tests for local file loading.
- Create `tests/test_synthetic.py`: unit tests for clean and contaminated record generation.
- Create `tests/test_detectors.py`: unit tests for obvious detector behavior.
- Create `tests/test_evaluation.py`: unit tests for metric output shape.
- Modify `README.md`: replace promise-only structure with runnable setup, lab flow, and the local/traditional ML teaching point.
- Modify `.gitignore` if present at repo root from this directory's perspective; add `outputs/`, `.pytest_cache/`, `__pycache__/`, and notebook checkpoint patterns if missing.

## Task 1: Project Skeleton and Dependencies

**Files:**
- Create: `requirements.txt`
- Create: `src/survey_contamination/__init__.py`
- Modify: `.gitignore`

- [ ] **Step 1: Create package directories**

Run:

```bash
mkdir -p src/survey_contamination scripts notebooks tests
```

Expected: directories exist.

- [ ] **Step 2: Add dependencies**

Create `requirements.txt` with:

```text
pandas>=2.2
numpy>=1.26
scikit-learn>=1.4
matplotlib>=3.8
seaborn>=0.13
jupyter>=1.0
pytest>=8.0
```

- [ ] **Step 3: Add package initializer**

Create `src/survey_contamination/__init__.py` with:

```python
"""Utilities for the survey contamination detection lab."""

__version__ = "0.1.0"
```

- [ ] **Step 4: Update generated-file ignores**

If `.gitignore` exists at the repository root, append these patterns if absent:

```gitignore
outputs/
__pycache__/
.pytest_cache/
.ipynb_checkpoints/
```

If no `.gitignore` is present in this lab directory, create one with those four patterns.

- [ ] **Step 5: Verify skeleton**

Run:

```bash
python -c "import pathlib; assert pathlib.Path('src/survey_contamination/__init__.py').exists()"
```

Expected: exit code 0.

- [ ] **Step 6: Commit skeleton**

Run:

```bash
git add requirements.txt src/survey_contamination/__init__.py .gitignore
git commit -m "Add lab package skeleton"
```

Expected: commit succeeds.

## Task 2: Local Data Loaders

**Files:**
- Create: `tests/test_data.py`
- Create: `src/survey_contamination/data.py`

- [ ] **Step 1: Write failing data-loader tests**

Create `tests/test_data.py` with:

```python
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from survey_contamination.data import load_first_names, load_surnames


def test_load_surnames_normalizes_names(tmp_path: Path):
    path = tmp_path / "surnames.csv"
    path.write_text(
        "name,rank,count,prop100k,cum_prop100k,pctwhite,pctblack,pctapi,pctaian,pct2prace,pcthispanic\n"
        "SMITH,1,2442977,828.19,828.19,70.9,23.11,0.5,0.89,2.19,2.4\n"
        "garcia,2,1166120,395.32,3400.12,5.38,0.45,1.41,0.47,0.26,92.03\n"
    )

    surnames = load_surnames(path)

    assert list(surnames["name"]) == ["Smith", "Garcia"]
    assert list(surnames["count"]) == [2442977, 1166120]
    assert surnames["source"].eq("census_surname").all()


def test_load_first_names_reads_state_zip(tmp_path: Path):
    zip_path = tmp_path / "namesbystate.zip"
    with ZipFile(zip_path, "w") as archive:
        archive.writestr("NY.TXT", "NY,F,2020,Olivia,1200\nNY,M,2020,Liam,1100\n")
        archive.writestr("CA.TXT", "CA,F,2020,Olivia,1300\nCA,M,2019,Noah,900\n")

    first_names = load_first_names(zip_path)

    assert set(first_names["name"]) == {"Olivia", "Liam", "Noah"}
    assert int(first_names.loc[first_names["name"] == "Olivia", "count"].iloc[0]) == 2500
    assert first_names["source"].eq("ssa_first_name").all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
PYTHONPATH=src python -m pytest tests/test_data.py -v
```

Expected: FAIL because `survey_contamination.data` does not exist.

- [ ] **Step 3: Implement data loaders**

Create `src/survey_contamination/data.py` with:

```python
from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SURNAME_PATH = PACKAGE_ROOT / "data" / "Names_2010Census.csv"
DEFAULT_FIRST_NAME_ZIP = PACKAGE_ROOT / "data" / "namesbystate.zip"


def _require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Expected data file at {path}")
    return path


def normalize_person_name(value: str) -> str:
    return str(value).strip().title()


def load_surnames(path: str | Path = DEFAULT_SURNAME_PATH, top_n: int | None = None) -> pd.DataFrame:
    data_path = _require_file(Path(path))
    frame = pd.read_csv(data_path)
    required = {"name", "rank", "count"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Surname file is missing columns: {sorted(missing)}")

    result = frame.loc[:, ["name", "rank", "count"]].copy()
    result["name"] = result["name"].map(normalize_person_name)
    result["source"] = "census_surname"
    result = result.sort_values(["rank", "name"], ascending=[True, True]).reset_index(drop=True)
    if top_n is not None:
        result = result.head(top_n).copy()
    return result


def load_first_names(path: str | Path = DEFAULT_FIRST_NAME_ZIP, top_n: int | None = None) -> pd.DataFrame:
    data_path = _require_file(Path(path))
    rows: list[pd.DataFrame] = []
    columns = ["state", "sex", "year", "name", "count"]

    with ZipFile(data_path) as archive:
        for member in archive.namelist():
            if not member.endswith(".TXT"):
                continue
            with archive.open(member) as file_obj:
                rows.append(pd.read_csv(file_obj, names=columns))

    if not rows:
        raise ValueError(f"No SSA state TXT files found in {data_path}")

    frame = pd.concat(rows, ignore_index=True)
    frame["name"] = frame["name"].map(normalize_person_name)
    result = (
        frame.groupby("name", as_index=False)["count"]
        .sum()
        .sort_values(["count", "name"], ascending=[False, True])
        .reset_index(drop=True)
    )
    result.insert(1, "rank", range(1, len(result) + 1))
    result["source"] = "ssa_first_name"
    if top_n is not None:
        result = result.head(top_n).copy()
    return result
```

- [ ] **Step 4: Run data-loader tests**

Run:

```bash
PYTHONPATH=src python -m pytest tests/test_data.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit data loaders**

Run:

```bash
git add tests/test_data.py src/survey_contamination/data.py
git commit -m "Add local name data loaders"
```

Expected: commit succeeds.

## Task 3: Synthetic Business and Contamination Generation

**Files:**
- Create: `tests/test_synthetic.py`
- Create: `src/survey_contamination/synthetic.py`

- [ ] **Step 1: Write failing synthetic-data tests**

Create `tests/test_synthetic.py` with:

```python
import pandas as pd

from survey_contamination.synthetic import generate_labeled_business_names


def test_generate_labeled_business_names_has_expected_columns():
    first_names = pd.DataFrame({"name": ["John", "Maria"], "count": [100, 80]})
    surnames = pd.DataFrame({"name": ["Smith", "Garcia"], "count": [200, 150]})

    frame = generate_labeled_business_names(
        first_names=first_names,
        surnames=surnames,
        n_records=100,
        contamination_rate=0.1,
        seed=7,
    )

    assert set(frame.columns) == {"record_id", "business_name", "is_contaminated", "pattern"}
    assert len(frame) == 100
    assert frame["record_id"].is_unique
    assert frame["is_contaminated"].sum() == 10


def test_clean_business_examples_include_business_signals():
    first_names = pd.DataFrame({"name": ["John"], "count": [100]})
    surnames = pd.DataFrame({"name": ["Smith"], "count": [200]})

    frame = generate_labeled_business_names(
        first_names=first_names,
        surnames=surnames,
        n_records=20,
        contamination_rate=0.0,
        seed=3,
    )

    assert not frame["is_contaminated"].any()
    assert frame["business_name"].str.contains("LLC|Inc|Company|Services|Group|Associates", regex=True).any()
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
PYTHONPATH=src python -m pytest tests/test_synthetic.py -v
```

Expected: FAIL because `survey_contamination.synthetic` does not exist.

- [ ] **Step 3: Implement synthetic-data generator**

Create `src/survey_contamination/synthetic.py` with:

```python
from __future__ import annotations

import math

import numpy as np
import pandas as pd


BUSINESS_ROOTS = [
    "Northstar",
    "Blue Ridge",
    "Riverbend",
    "Summit",
    "Keystone",
    "Pioneer",
    "Cedar Valley",
    "Maple Street",
    "Harbor",
    "Prairie",
]
INDUSTRY_TERMS = [
    "Analytics",
    "Construction",
    "Logistics",
    "Consulting",
    "Dental",
    "Manufacturing",
    "Insurance",
    "Retail",
    "Survey Research",
    "Software",
]
BUSINESS_SUFFIXES = ["LLC", "Inc", "Company", "Services", "Group", "Associates"]
CONTAMINATION_PATTERNS = ["first_last", "first_middle_last", "initial_last", "last_first"]


def _weighted_choice(rng: np.random.Generator, frame: pd.DataFrame) -> str:
    weights = frame["count"].to_numpy(dtype=float)
    probabilities = weights / weights.sum()
    return str(rng.choice(frame["name"].to_numpy(), p=probabilities))


def _make_clean_business_name(rng: np.random.Generator) -> tuple[str, str]:
    pattern = str(rng.choice(["root_industry_suffix", "owner_industry_suffix", "root_suffix"]))
    if pattern == "root_industry_suffix":
        return (
            f"{rng.choice(BUSINESS_ROOTS)} {rng.choice(INDUSTRY_TERMS)} {rng.choice(BUSINESS_SUFFIXES)}",
            pattern,
        )
    if pattern == "owner_industry_suffix":
        return (
            f"{rng.choice(BUSINESS_ROOTS)} {rng.choice(INDUSTRY_TERMS)} {rng.choice(BUSINESS_SUFFIXES)}",
            pattern,
        )
    return (f"{rng.choice(BUSINESS_ROOTS)} {rng.choice(BUSINESS_SUFFIXES)}", pattern)


def _make_contaminated_name(
    rng: np.random.Generator,
    first_names: pd.DataFrame,
    surnames: pd.DataFrame,
) -> tuple[str, str]:
    first = _weighted_choice(rng, first_names)
    middle = _weighted_choice(rng, first_names)
    last = _weighted_choice(rng, surnames)
    pattern = str(rng.choice(CONTAMINATION_PATTERNS))

    if pattern == "first_middle_last":
        return f"{first} {middle[0]}. {last}", pattern
    if pattern == "initial_last":
        return f"{first[0]}. {last}", pattern
    if pattern == "last_first":
        return f"{last}, {first}", pattern
    return f"{first} {last}", pattern


def generate_labeled_business_names(
    first_names: pd.DataFrame,
    surnames: pd.DataFrame,
    n_records: int = 10_000,
    contamination_rate: float = 0.005,
    seed: int = 42,
) -> pd.DataFrame:
    if n_records <= 0:
        raise ValueError("n_records must be positive")
    if not 0 <= contamination_rate <= 1:
        raise ValueError("contamination_rate must be between 0 and 1")

    rng = np.random.default_rng(seed)
    contaminated_count = int(math.floor(n_records * contamination_rate))
    contaminated_ids = set(rng.choice(n_records, size=contaminated_count, replace=False).tolist())

    rows: list[dict[str, object]] = []
    for record_id in range(n_records):
        if record_id in contaminated_ids:
            name, pattern = _make_contaminated_name(rng, first_names, surnames)
            is_contaminated = True
        else:
            name, pattern = _make_clean_business_name(rng)
            is_contaminated = False
        rows.append(
            {
                "record_id": record_id,
                "business_name": name,
                "is_contaminated": is_contaminated,
                "pattern": pattern,
            }
        )

    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run synthetic-data tests**

Run:

```bash
PYTHONPATH=src python -m pytest tests/test_synthetic.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit synthetic generator**

Run:

```bash
git add tests/test_synthetic.py src/survey_contamination/synthetic.py
git commit -m "Add synthetic contamination generator"
```

Expected: commit succeeds.

## Task 4: Feature Engineering and Detectors

**Files:**
- Create: `tests/test_detectors.py`
- Create: `src/survey_contamination/features.py`
- Create: `src/survey_contamination/detectors.py`

- [ ] **Step 1: Write failing detector tests**

Create `tests/test_detectors.py` with:

```python
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
```

- [ ] **Step 2: Run detector tests to verify they fail**

Run:

```bash
PYTHONPATH=src python -m pytest tests/test_detectors.py -v
```

Expected: FAIL because detector modules do not exist.

- [ ] **Step 3: Implement feature extraction**

Create `src/survey_contamination/features.py` with:

```python
from __future__ import annotations

import re
from collections.abc import Iterable

import pandas as pd


TOKEN_RE = re.compile(r"[A-Za-z]+")
BUSINESS_TERMS = {
    "associates",
    "company",
    "consulting",
    "group",
    "inc",
    "insurance",
    "llc",
    "logistics",
    "manufacturing",
    "services",
}


def tokenize(value: str) -> list[str]:
    return TOKEN_RE.findall(str(value))


def featurize_names(
    values: Iterable[str],
    first_names: set[str],
    surnames: set[str],
) -> pd.DataFrame:
    first_lookup = {name.lower() for name in first_names}
    surname_lookup = {name.lower() for name in surnames}

    rows: list[dict[str, float]] = []
    for value in values:
        tokens = tokenize(value)
        lowered = [token.lower() for token in tokens]
        token_count = len(tokens)
        first_matches = sum(token in first_lookup for token in lowered)
        surname_matches = sum(token in surname_lookup for token in lowered)
        business_matches = sum(token in BUSINESS_TERMS for token in lowered)
        rows.append(
            {
                "token_count": float(token_count),
                "char_count": float(len(str(value))),
                "first_name_matches": float(first_matches),
                "surname_matches": float(surname_matches),
                "name_match_ratio": float((first_matches + surname_matches) / max(token_count, 1)),
                "business_term_matches": float(business_matches),
                "has_comma": float("," in str(value)),
                "has_period": float("." in str(value)),
            }
        )
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Implement detectors**

Create `src/survey_contamination/detectors.py` with:

```python
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from survey_contamination.features import BUSINESS_TERMS, featurize_names, tokenize


class NameDictionaryDetector:
    name = "name_dictionary"

    def __init__(self, first_names: set[str], surnames: set[str], threshold: float = 0.75):
        self.first_names = {name.lower() for name in first_names}
        self.surnames = {name.lower() for name in surnames}
        self.threshold = threshold

    def score(self, values: Iterable[str]) -> np.ndarray:
        scores: list[float] = []
        for value in values:
            tokens = [token.lower() for token in tokenize(value)]
            if not tokens:
                scores.append(0.0)
                continue
            first_matches = sum(token in self.first_names for token in tokens)
            surname_matches = sum(token in self.surnames for token in tokens)
            has_first = first_matches > 0
            has_surname = surname_matches > 0
            ratio = (first_matches + surname_matches) / len(tokens)
            scores.append(1.0 if has_first and has_surname else ratio * 0.5)
        return np.array(scores, dtype=float)

    def predict(self, values: Iterable[str]) -> np.ndarray:
        return self.score(values) >= self.threshold


class BusinessSuffixHeuristicDetector(NameDictionaryDetector):
    name = "business_suffix_heuristic"

    def score(self, values: Iterable[str]) -> np.ndarray:
        base_scores = super().score(values)
        adjusted: list[float] = []
        for value, score in zip(values, base_scores, strict=False):
            tokens = {token.lower() for token in tokenize(value)}
            if tokens.intersection(BUSINESS_TERMS):
                adjusted.append(min(float(score), 0.25))
            else:
                adjusted.append(float(score))
        return np.array(adjusted, dtype=float)


class FeatureClassifierDetector:
    name = "feature_classifier"

    def __init__(self, first_names: set[str], surnames: set[str], threshold: float = 0.5):
        self.first_names = first_names
        self.surnames = surnames
        self.threshold = threshold
        self.model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")

    def fit(self, values: Iterable[str], labels: Iterable[bool]) -> "FeatureClassifierDetector":
        features = featurize_names(values, self.first_names, self.surnames)
        self.model.fit(features, list(labels))
        return self

    def score(self, values: Iterable[str]) -> np.ndarray:
        features = featurize_names(values, self.first_names, self.surnames)
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(features)[:, 1]
        return self.model.predict(features).astype(float)

    def predict(self, values: Iterable[str]) -> np.ndarray:
        return self.score(values) >= self.threshold


class SpacyPersonDetector:
    name = "spacy_person"

    def __init__(self, model_name: str = "en_core_web_lg", threshold: float = 0.5):
        try:
            import spacy
        except ImportError as exc:
            raise RuntimeError("spaCy is optional and is not installed.") from exc
        try:
            self.nlp = spacy.load(model_name)
        except OSError as exc:
            raise RuntimeError(f"spaCy model {model_name!r} is not installed.") from exc
        self.threshold = threshold

    def score(self, values: Iterable[str]) -> np.ndarray:
        scores: list[float] = []
        for doc in self.nlp.pipe([str(value) for value in values]):
            has_person = any(ent.label_ == "PERSON" for ent in doc.ents)
            scores.append(1.0 if has_person else 0.0)
        return np.array(scores, dtype=float)

    def predict(self, values: Iterable[str]) -> np.ndarray:
        return self.score(values) >= self.threshold


def build_name_sets(first_names: pd.DataFrame, surnames: pd.DataFrame, top_n: int = 5_000) -> tuple[set[str], set[str]]:
    first_lookup = set(first_names.head(top_n)["name"].astype(str))
    surname_lookup = set(surnames.head(top_n)["name"].astype(str))
    return first_lookup, surname_lookup
```

- [ ] **Step 5: Run detector tests**

Run:

```bash
PYTHONPATH=src python -m pytest tests/test_detectors.py -v
```

Expected: 3 passed.

- [ ] **Step 6: Commit detectors**

Run:

```bash
git add tests/test_detectors.py src/survey_contamination/features.py src/survey_contamination/detectors.py
git commit -m "Add contamination detectors"
```

Expected: commit succeeds.

## Task 5: Evaluation Utilities

**Files:**
- Create: `tests/test_evaluation.py`
- Create: `src/survey_contamination/evaluation.py`

- [ ] **Step 1: Write failing evaluation tests**

Create `tests/test_evaluation.py` with:

```python
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
```

- [ ] **Step 2: Run evaluation tests to verify they fail**

Run:

```bash
PYTHONPATH=src python -m pytest tests/test_evaluation.py -v
```

Expected: FAIL because `survey_contamination.evaluation` does not exist.

- [ ] **Step 3: Implement evaluation utilities**

Create `src/survey_contamination/evaluation.py` with:

```python
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
```

- [ ] **Step 4: Run evaluation tests**

Run:

```bash
PYTHONPATH=src python -m pytest tests/test_evaluation.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit evaluation utilities**

Run:

```bash
git add tests/test_evaluation.py src/survey_contamination/evaluation.py
git commit -m "Add evaluation utilities"
```

Expected: commit succeeds.

## Task 6: Reproducible Pipeline Scripts

**Files:**
- Create: `scripts/01_data_preparation.py`
- Create: `scripts/02_generate_contamination.py`
- Create: `scripts/03_run_detection.py`
- Create: `scripts/04_evaluate.py`

- [ ] **Step 1: Create data-preparation script**

Create `scripts/01_data_preparation.py` with:

```python
from __future__ import annotations

from pathlib import Path

from survey_contamination.data import load_first_names, load_surnames


OUTPUT_DIR = Path("outputs")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    surnames = load_surnames(top_n=10_000)
    first_names = load_first_names(top_n=10_000)
    surnames.to_csv(OUTPUT_DIR / "surnames_top.csv", index=False)
    first_names.to_csv(OUTPUT_DIR / "first_names_top.csv", index=False)
    print(f"Wrote {len(surnames):,} surnames to {OUTPUT_DIR / 'surnames_top.csv'}")
    print(f"Wrote {len(first_names):,} first names to {OUTPUT_DIR / 'first_names_top.csv'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Create contamination-generation script**

Create `scripts/02_generate_contamination.py` with:

```python
from __future__ import annotations

from pathlib import Path

from survey_contamination.data import load_first_names, load_surnames
from survey_contamination.synthetic import generate_labeled_business_names


OUTPUT_DIR = Path("outputs")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    surnames = load_surnames(top_n=5_000)
    first_names = load_first_names(top_n=5_000)
    frame = generate_labeled_business_names(
        first_names=first_names,
        surnames=surnames,
        n_records=10_000,
        contamination_rate=0.005,
        seed=42,
    )
    output_path = OUTPUT_DIR / "synthetic_business_names.csv"
    frame.to_csv(output_path, index=False)
    print(f"Wrote {len(frame):,} labeled records to {output_path}")
    print(f"Contaminated rows: {int(frame['is_contaminated'].sum()):,}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Create detector-running script**

Create `scripts/03_run_detection.py` with:

```python
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
```

- [ ] **Step 4: Create evaluation script**

Create `scripts/04_evaluate.py` with:

```python
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from survey_contamination.evaluation import evaluate_detector_predictions, summarize_error_examples


OUTPUT_DIR = Path("outputs")


def main() -> None:
    input_path = OUTPUT_DIR / "detector_scores.csv"
    if not input_path.exists():
        raise FileNotFoundError("Run scripts/03_run_detection.py before this script.")

    scores = pd.read_csv(input_path)
    metrics = pd.DataFrame(
        [
            evaluate_detector_predictions(group, detector_name=str(detector))
            for detector, group in scores.groupby("detector")
        ]
    ).sort_values("f1", ascending=False)
    metrics_path = OUTPUT_DIR / "detector_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    errors = pd.concat(
        [
            summarize_error_examples(group.assign(detector=str(detector)), max_examples=10)
            for detector, group in scores.groupby("detector")
        ],
        ignore_index=True,
    )
    errors_path = OUTPUT_DIR / "error_examples.csv"
    errors.to_csv(errors_path, index=False)

    plt.figure(figsize=(8, 5))
    plot_data = metrics.melt(
        id_vars="detector",
        value_vars=["precision", "recall", "f1"],
        var_name="metric",
        value_name="value",
    )
    sns.barplot(data=plot_data, x="detector", y="value", hue="metric")
    plt.ylim(0, 1)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plot_path = OUTPUT_DIR / "detector_metrics.png"
    plt.savefig(plot_path, dpi=150)

    print(f"Wrote metrics to {metrics_path}")
    print(f"Wrote error examples to {errors_path}")
    print(f"Wrote metric plot to {plot_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run scripts in order**

Run:

```bash
PYTHONPATH=src python scripts/01_data_preparation.py
PYTHONPATH=src python scripts/02_generate_contamination.py
PYTHONPATH=src python scripts/03_run_detection.py
PYTHONPATH=src python scripts/04_evaluate.py
```

Expected:

- `outputs/surnames_top.csv` exists.
- `outputs/first_names_top.csv` exists.
- `outputs/synthetic_business_names.csv` exists.
- `outputs/detector_scores.csv` exists.
- `outputs/detector_metrics.csv` exists.
- `outputs/error_examples.csv` exists.
- `outputs/detector_metrics.png` exists.

- [ ] **Step 6: Commit scripts**

Run:

```bash
git add scripts/01_data_preparation.py scripts/02_generate_contamination.py scripts/03_run_detection.py scripts/04_evaluate.py
git commit -m "Add reproducible lab pipeline scripts"
```

Expected: commit succeeds.

## Task 7: Teaching Notebook

**Files:**
- Create: `notebooks/contamination_detection_lab.ipynb`

- [ ] **Step 1: Create notebook with executable cells**

Create `notebooks/contamination_detection_lab.ipynb` using `nbformat` or Jupyter. Include these sections in order:

```markdown
# Survey Processing Contamination Detection

This lab demonstrates a local, traditional ML approach to a targeted survey data quality problem. The goal is not to reach for an LLM, but to show how rules, dictionaries, and standard Python ML tools can solve the problem privately and reproducibly.
```

```python
from pathlib import Path
import sys

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT / "src"))
```

```python
from survey_contamination.data import load_first_names, load_surnames
from survey_contamination.detectors import (
    BusinessSuffixHeuristicDetector,
    FeatureClassifierDetector,
    NameDictionaryDetector,
    build_name_sets,
)
from survey_contamination.evaluation import evaluate_detector_predictions, summarize_error_examples
from survey_contamination.synthetic import generate_labeled_business_names
```

```python
surnames = load_surnames(top_n=5000)
first_names = load_first_names(top_n=5000)
surnames.head(), first_names.head()
```

```python
frame = generate_labeled_business_names(first_names, surnames, n_records=10000, contamination_rate=0.005, seed=42)
frame["is_contaminated"].value_counts()
```

```python
first_lookup, surname_lookup = build_name_sets(first_names, surnames)
detectors = [
    NameDictionaryDetector(first_lookup, surname_lookup),
    BusinessSuffixHeuristicDetector(first_lookup, surname_lookup),
]
```

```python
from sklearn.model_selection import train_test_split
import pandas as pd

train, test = train_test_split(frame, test_size=0.35, random_state=42, stratify=frame["is_contaminated"])
classifier = FeatureClassifierDetector(first_lookup, surname_lookup)
classifier.fit(train["business_name"], train["is_contaminated"])
detectors.append(classifier)

scored = []
for detector in detectors:
    output = test.copy()
    output["detector"] = detector.name
    output["score"] = detector.score(output["business_name"])
    output["predicted"] = detector.predict(output["business_name"])
    scored.append(output)
scores = pd.concat(scored, ignore_index=True)
```

```python
metrics = pd.DataFrame(
    evaluate_detector_predictions(group, detector_name=name)
    for name, group in scores.groupby("detector")
).sort_values("f1", ascending=False)
metrics
```

```python
errors = pd.concat(
    summarize_error_examples(group.assign(detector=name), max_examples=5)
    for name, group in scores.groupby("detector")
)
errors[["detector", "error_type", "business_name", "is_contaminated", "predicted", "score"]]
```

Add markdown interpretation cells that explain:

- Low contamination rates make false positives operationally expensive.
- Local processing keeps respondent/business data private.
- The feature classifier is a traditional ML baseline, not an LLM.
- spaCy can be explored optionally, but the core lab works without it.

- [ ] **Step 2: Execute notebook**

Run:

```bash
PYTHONPATH=src jupyter nbconvert --to notebook --execute notebooks/contamination_detection_lab.ipynb --output contamination_detection_lab.executed.ipynb --output-dir outputs
```

Expected: exit code 0 and `outputs/contamination_detection_lab.executed.ipynb` exists.

- [ ] **Step 3: Commit notebook**

Run:

```bash
git add notebooks/contamination_detection_lab.ipynb
git commit -m "Add contamination detection teaching notebook"
```

Expected: commit succeeds.

## Task 8: README and Lab Instructions

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Rewrite README around the runnable lab**

Update `README.md` to include these sections:

```markdown
# Survey Processing Contamination Detection

A practical lab for detecting personal names accidentally entered in business name fields using local rules, standard Python packages, and traditional machine learning.

## Why Not Start With an LLM?

This lab is intentionally not an LLM lab. The problem has structured clues, local reference data, measurable error trade-offs, and privacy considerations. Those properties make it a strong fit for deterministic rules and traditional ML before considering hosted AI services.
```

Include setup:

```markdown
## Setup

```bash
conda create -n survey-contamination python=3.11
conda activate survey-contamination
pip install -r requirements.txt
```
```

Include script workflow:

```markdown
## Run the Pipeline

```bash
PYTHONPATH=src python scripts/01_data_preparation.py
PYTHONPATH=src python scripts/02_generate_contamination.py
PYTHONPATH=src python scripts/03_run_detection.py
PYTHONPATH=src python scripts/04_evaluate.py
```
```

Include notebook workflow:

```markdown
## Run the Notebook

```bash
jupyter notebook notebooks/contamination_detection_lab.ipynb
```
```

Include optional spaCy:

```markdown
## Optional spaCy Baseline

The core lab does not require spaCy. To try the optional NER detector:

```bash
pip install spacy
python -m spacy download en_core_web_lg
```
```

- [ ] **Step 2: Check README command blocks**

Run:

```bash
rg -n "Why Not Start With an LLM|Run the Pipeline|Optional spaCy" README.md
```

Expected: all three headings are found.

- [ ] **Step 3: Commit README**

Run:

```bash
git add README.md
git commit -m "Document runnable local ML lab"
```

Expected: commit succeeds.

## Task 9: Full Verification

**Files:**
- No new files unless fixes are needed.

- [ ] **Step 1: Run all tests**

Run:

```bash
PYTHONPATH=src python -m pytest -v
```

Expected: all tests pass.

- [ ] **Step 2: Run complete script pipeline**

Run:

```bash
PYTHONPATH=src python scripts/01_data_preparation.py
PYTHONPATH=src python scripts/02_generate_contamination.py
PYTHONPATH=src python scripts/03_run_detection.py
PYTHONPATH=src python scripts/04_evaluate.py
```

Expected: all commands exit 0 and print output paths.

- [ ] **Step 3: Verify final output files**

Run:

```bash
python - <<'PY'
from pathlib import Path

required = [
    "outputs/surnames_top.csv",
    "outputs/first_names_top.csv",
    "outputs/synthetic_business_names.csv",
    "outputs/detector_scores.csv",
    "outputs/detector_metrics.csv",
    "outputs/error_examples.csv",
    "outputs/detector_metrics.png",
]
missing = [path for path in required if not Path(path).exists()]
if missing:
    raise SystemExit(f"Missing outputs: {missing}")
print("All expected outputs exist.")
PY
```

Expected: prints `All expected outputs exist.`

- [ ] **Step 4: Inspect git status**

Run:

```bash
git status --short
```

Expected: only intentionally untracked local files outside this lab or generated ignored files are present.

- [ ] **Step 5: Commit final fixes if needed**

If verification required code or documentation fixes, run:

```bash
git add README.md requirements.txt .gitignore src/survey_contamination scripts notebooks tests
git commit -m "Fix lab verification issues"
```

Expected: commit succeeds if there were fixes; no commit is created if there were no fixes.

## Self-Review

Spec coverage:

- Local Census surname loading is covered by Task 2 and Task 6.
- SSA first-name zip loading is covered by Task 2 and Task 6.
- Synthetic clean businesses and contamination injection are covered by Task 3.
- Rule-based, heuristic, ML, and optional spaCy detectors are covered by Task 4 and Task 6.
- Evaluation metrics, error examples, and plots are covered by Task 5 and Task 6.
- Notebook teaching flow is covered by Task 7.
- README setup and local/traditional ML positioning are covered by Task 8.
- Tests and full verification are covered by Tasks 2 through 5 and Task 9.

Red-flag scan:

- Deferred-work markers were checked and removed.
- Optional spaCy behavior is concrete: skip with a clear message when unavailable.

Type consistency:

- Shared detector methods are `score(values)` and `predict(values)`.
- Synthetic labels use `is_contaminated`.
- Script outputs use stable paths under `outputs/`.
