import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot precision, recall, and ROC AUC from a corruption sweep CSV."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("corruption_results.csv"),
        help="CSV produced by sweep_corruption.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("corruption_metrics.png"),
        help="Where to save the resulting plot.",
    )
    return parser.parse_args()


def load_results(path: Path):
    fractions, precision, recall, roc_auc = [], [], [], []
    with path.open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            fractions.append(float(row["corrupt_frac"]))
            precision.append(float(row["precision"]))
            recall.append(float(row["recall"]))
            roc_auc.append(float(row["roc_auc"]))
    if not fractions:
        raise ValueError(f"No rows found in {path}")
    return fractions, precision, recall, roc_auc


def plot_metrics(fractions, precision, recall, roc_auc, output: Path):
    fig, axes = plt.subplots(3, 1, figsize=(6, 8), sharex=True)
    metrics = [
        ("Precision", precision),
        ("Recall", recall),
        ("ROC AUC", roc_auc),
    ]

    for ax, (label, values) in zip(axes, metrics):
        ax.plot(fractions, values, marker="o")
        ax.set_ylabel(label)
        ax.grid(True, linestyle="--", alpha=0.4)

    axes[-1].set_xlabel("Corruption fraction")
    fig.suptitle("Impact of MNIST label corruption")
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    print(f"[plot] Saved figure to {output}")


def main():
    args = parse_args()
    fractions, precision, recall, roc_auc = load_results(args.input)
    plot_metrics(fractions, precision, recall, roc_auc, args.output)


if __name__ == "__main__":
    main()
