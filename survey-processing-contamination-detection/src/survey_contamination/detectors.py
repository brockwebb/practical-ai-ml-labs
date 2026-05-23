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
        value_list = list(values)
        base_scores = super().score(value_list)
        adjusted: list[float] = []
        for value, score in zip(value_list, base_scores, strict=False):
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
