from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
import pandas as pd


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from survey_contamination.evaluation import evaluate_detector_predictions, summarize_error_examples


OUTPUT_DIR = PROJECT_ROOT / "outputs"


def main() -> None:
    input_path = OUTPUT_DIR / "detector_scores.csv"
    if not input_path.exists():
        raise FileNotFoundError("Run scripts/03_run_detection.py before this script.")

    try:
        scores = pd.read_csv(input_path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError("Detector scores are empty. Run scripts/03_run_detection.py again.") from exc

    if scores.empty:
        raise ValueError("Detector scores are empty. Run scripts/03_run_detection.py again.")

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
