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
