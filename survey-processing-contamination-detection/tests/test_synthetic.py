import pandas as pd
import pytest

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


def test_clean_business_examples_include_owner_style_names():
    first_names = pd.DataFrame({"name": ["John"], "count": [100]})
    surnames = pd.DataFrame({"name": ["Smith", "Garcia"], "count": [200, 150]})

    frame = generate_labeled_business_names(
        first_names=first_names,
        surnames=surnames,
        n_records=200,
        contamination_rate=0.0,
        seed=11,
    )

    owner_style = frame[frame["pattern"] == "owner_industry_suffix"]

    assert not owner_style.empty
    assert not owner_style["is_contaminated"].any()
    assert owner_style["business_name"].str.contains(r"^(?:Smith|Garcia) ").all()


def test_invalid_weighted_input_missing_count_raises_value_error():
    first_names = pd.DataFrame({"name": ["John"]})
    surnames = pd.DataFrame({"name": ["Smith"], "count": [200]})

    with pytest.raises(ValueError, match="count"):
        generate_labeled_business_names(
            first_names=first_names,
            surnames=surnames,
            n_records=10,
            contamination_rate=1.0,
            seed=5,
        )


def test_invalid_weighted_input_all_zero_counts_raises_value_error():
    first_names = pd.DataFrame({"name": ["John"], "count": [0]})
    surnames = pd.DataFrame({"name": ["Smith"], "count": [200]})

    with pytest.raises(ValueError, match="positive"):
        generate_labeled_business_names(
            first_names=first_names,
            surnames=surnames,
            n_records=10,
            contamination_rate=1.0,
            seed=5,
        )
