import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List


def generate_fractions(max_frac: float, step: float) -> List[float]:
    fractions = []
    current = 0.0
    while current <= max_frac + 1e-9:  # accommodate floating error
        fractions.append(round(current, 4))
        current += step
    return fractions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep main_corrupted.py over several corruption fractions and store metrics."
    )
    parser.add_argument(
        "--fractions",
        type=float,
        nargs="+",
        default=None,
        help="Explicit list of corruption fractions (each between 0 and 0.9).",
    )
    parser.add_argument(
        "--max-frac",
        type=float,
        default=0.9,
        help="Largest corruption fraction to try when --fractions is not provided.",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=0.1,
        help="Step size when generating fractions (used only if --fractions omitted).",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs per run.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size per run.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate per run.")
    parser.add_argument("--data-dir", type=str, default="data", help="Where MNIST is stored/downloaded.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed forwarded to main_corrupted.py.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("corruption_results.csv"),
        help="CSV file that will contain the sweep metrics.",
    )
    return parser.parse_args()


def validate_fractions(fractions: Iterable[float]) -> List[float]:
    result = []
    for frac in fractions:
        if not (0.0 <= frac <= 0.9):
            raise ValueError(f"Fraction {frac} is outside the supported [0, 0.9] range.")
        result.append(round(frac, 4))
    return result


def run_single_fraction(fraction: float, args: argparse.Namespace) -> dict:
    cmd = [
        sys.executable,
        "main_corrupted.py",
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--data-dir",
        args.data_dir,
        "--corrupt-frac",
        str(fraction),
        "--seed",
        str(args.seed),
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True, check=True)
    metrics_line = extract_metrics_line(completed.stdout)
    metrics = parse_metrics(metrics_line)
    metrics.update({"corrupt_frac": fraction})
    print(
        f"[sweep] frac={fraction:.2f} | "
        f"acc={metrics['test_acc']*100:5.2f}% | "
        f"precision={metrics['precision']:.3f} | "
        f"recall={metrics['recall']:.3f} | "
        f"roc_auc={metrics['roc_auc']:.3f}"
    )
    return metrics


def extract_metrics_line(stdout: str) -> str:
    for line in reversed(stdout.strip().splitlines()):
        if line.startswith("Test loss"):
            return line.strip()
    raise RuntimeError("Failed to find metrics line in main_corrupted.py output:\n" + stdout)


def parse_metrics(line: str) -> dict:
    # Expected format:
    # Test loss 0.431, acc 93.04% | precision 0.930 | recall 0.930 | ROC AUC 0.996
    parts = [segment.strip() for segment in line.split("|")]
    loss_part, acc_part = [segment.strip() for segment in parts[0].split(",")]
    precision_part, recall_part, roc_auc_part = parts[1:]

    metrics = {
        "test_loss": float(loss_part.split()[-1]),
        "test_acc": float(acc_part.replace("acc", "").replace("%", "").strip()) / 100.0,
        "precision": float(precision_part.split()[-1]),
        "recall": float(recall_part.split()[-1]),
        "roc_auc": float(roc_auc_part.split()[-1]),
    }
    return metrics


def write_results(rows: List[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["corrupt_frac", "test_loss", "test_acc", "precision", "recall", "roc_auc"]
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fieldnames})


def main():
    args = parse_args()
    if args.fractions is None:
        fractions = generate_fractions(args.max_frac, args.step)
    else:
        fractions = args.fractions
    fractions = validate_fractions(fractions)

    rows = []
    for frac in fractions:
        row = run_single_fraction(frac, args)
        rows.append(row)

    write_results(rows, args.output)
    print(f"[sweep] Saved {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
