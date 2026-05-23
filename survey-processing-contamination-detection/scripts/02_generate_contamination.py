from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from survey_contamination.synthetic import generate_labeled_business_names


OUTPUT_DIR = PROJECT_ROOT / "outputs"


def require_input(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run scripts/01_data_preparation.py before this script.")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    surnames_path = OUTPUT_DIR / "surnames_top.csv"
    first_names_path = OUTPUT_DIR / "first_names_top.csv"
    require_input(surnames_path)
    require_input(first_names_path)

    surnames = pd.read_csv(surnames_path).head(5_000)
    first_names = pd.read_csv(first_names_path).head(5_000)
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
