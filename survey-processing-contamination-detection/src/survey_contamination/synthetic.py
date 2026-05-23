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
    required_columns = {"name", "count"}
    missing_columns = required_columns - set(frame.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"name data is missing required column(s): {missing}")
    if frame.empty:
        raise ValueError("name data must contain at least one row")

    names = frame["name"]
    if names.isna().any() or names.astype(str).str.strip().eq("").any():
        raise ValueError("name data must include non-empty names")

    try:
        weights = pd.to_numeric(frame["count"], errors="raise").to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("name data count values must be numeric") from exc

    if not np.isfinite(weights).all():
        raise ValueError("name data count values must be finite")
    if (weights < 0).any():
        raise ValueError("name data count values must not be negative")

    total_weight = weights.sum()
    if not np.isfinite(total_weight) or total_weight <= 0:
        raise ValueError("name data must have a positive finite total count weight")

    probabilities = weights / total_weight
    return str(rng.choice(names.to_numpy(), p=probabilities))


def _make_clean_business_name(rng: np.random.Generator, surnames: pd.DataFrame) -> tuple[str, str]:
    pattern = str(rng.choice(["root_industry_suffix", "owner_industry_suffix", "root_suffix"]))
    if pattern == "root_industry_suffix":
        return (
            f"{rng.choice(BUSINESS_ROOTS)} {rng.choice(INDUSTRY_TERMS)} {rng.choice(BUSINESS_SUFFIXES)}",
            pattern,
        )
    if pattern == "owner_industry_suffix":
        owner = _weighted_choice(rng, surnames)
        suffix = str(rng.choice(BUSINESS_SUFFIXES))
        if suffix == "Associates":
            return (f"{owner} & Associates", pattern)
        return (
            f"{owner} {rng.choice(INDUSTRY_TERMS)} {suffix}",
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
            name, pattern = _make_clean_business_name(rng, surnames)
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
