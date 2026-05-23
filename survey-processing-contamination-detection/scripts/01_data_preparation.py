from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from survey_contamination.data import load_first_names, load_surnames


OUTPUT_DIR = PROJECT_ROOT / "outputs"


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
