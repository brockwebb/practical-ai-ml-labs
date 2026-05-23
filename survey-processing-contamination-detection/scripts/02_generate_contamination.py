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
