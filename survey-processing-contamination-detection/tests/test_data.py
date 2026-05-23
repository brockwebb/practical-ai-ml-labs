from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import pytest

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


def test_load_surnames_excludes_aggregate_bucket_from_top_n(tmp_path: Path):
    path = tmp_path / "surnames.csv"
    path.write_text(
        "name,rank,count,prop100k,cum_prop100k,pctwhite,pctblack,pctapi,pctaian,pct2prace,pcthispanic\n"
        "All Other Names,0,29312001,9937.93,9937.93,0,0,0,0,0,0\n"
        "SMITH,1,2442977,828.19,828.19,70.9,23.11,0.5,0.89,2.19,2.4\n"
    )

    surnames = load_surnames(path, top_n=1)

    assert list(surnames["name"]) == ["Smith"]


def test_load_first_names_reads_state_zip(tmp_path: Path):
    zip_path = tmp_path / "namesbystate.zip"
    with ZipFile(zip_path, "w") as archive:
        archive.writestr("NY.TXT", "NY,F,2020,Olivia,1200\nNY,M,2020,Liam,1100\n")
        archive.writestr("CA.TXT", "CA,F,2020,Olivia,1300\nCA,M,2019,Noah,900\n")

    first_names = load_first_names(zip_path)

    assert set(first_names["name"]) == {"Olivia", "Liam", "Noah"}
    assert int(first_names.loc[first_names["name"] == "Olivia", "count"].iloc[0]) == 2500
    assert first_names["source"].eq("ssa_first_name").all()


def test_load_first_names_rejects_invalid_count_with_member_name(tmp_path: Path):
    zip_path = tmp_path / "namesbystate.zip"
    with ZipFile(zip_path, "w") as archive:
        archive.writestr("NY.TXT", "NY,F,2020,Olivia,not-a-number\n")

    with pytest.raises(ValueError, match="NY.TXT"):
        load_first_names(zip_path)
