from __future__ import annotations

import csv
from io import TextIOWrapper
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


def _read_ssa_member(archive: ZipFile, member: str) -> pd.DataFrame:
    columns = ["state", "sex", "year", "name", "count"]
    records: list[list[str]] = []

    with archive.open(member) as file_obj:
        text_file = TextIOWrapper(file_obj, encoding="utf-8", newline="")
        for line_number, row in enumerate(csv.reader(text_file), start=1):
            if len(row) != len(columns):
                raise ValueError(
                    f"Malformed SSA row in {member} at line {line_number}: "
                    f"expected {len(columns)} fields, found {len(row)}"
                )
            records.append(row)

    frame = pd.DataFrame(records, columns=columns)
    missing_name = frame["name"].isna() | frame["name"].str.strip().eq("")
    missing_count = frame["count"].isna() | frame["count"].str.strip().eq("")
    if missing_name.any() or missing_count.any():
        raise ValueError(f"SSA file {member} contains missing name or count values")

    numeric_count = pd.to_numeric(frame["count"], errors="coerce")
    if numeric_count.isna().any():
        raise ValueError(f"SSA file {member} contains non-numeric count values")

    frame["count"] = numeric_count.astype("int64")
    return frame


def load_surnames(path: str | Path = DEFAULT_SURNAME_PATH, top_n: int | None = None) -> pd.DataFrame:
    data_path = _require_file(Path(path))
    frame = pd.read_csv(data_path)
    required = {"name", "rank", "count"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Surname file is missing columns: {sorted(missing)}")

    frame = frame.loc[frame["rank"] != 0].copy()
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

    with ZipFile(data_path) as archive:
        for member in archive.namelist():
            if not member.endswith(".TXT"):
                continue
            rows.append(_read_ssa_member(archive, member))

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
