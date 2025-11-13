## Minimal PyTorch MNIST Example

This project shows a compact end-to-end MNIST classifier using PyTorch and `uv` for dependency management. It includes clean train/validation/test splits and runs entirely from `main.py`.

### Prerequisites

- Python 3.12 (handled automatically by `uv` if missing)
- `uv` CLI (`pip install uv` or see <https://github.com/astral-sh/uv>)
- CUDA 12.1 capable drivers if you want GPU acceleration (CPU works too)

### Setup

```bash
# stay inside the repo root
export UV_CACHE_DIR=.uv-cache             # keeps caches writable inside the repo
export UV_TORCH_BACKEND=cu121             # pick your preferred backend (cpu, cu121, rocm*, …)
uv sync                                   # creates .venv and installs torch/vision with the backend above
```

### Run training

```bash
UV_CACHE_DIR=.uv-cache uv run python main.py --epochs 5 --batch-size 128
```

Flags:

- `--epochs`: number of training passes (default 5)
- `--batch-size`: mini-batch size (default 128)
- `--lr`: learning rate (default 1e-3)
- `--data-dir`: where MNIST will be cached or searched (`data` by default). If your machine cannot reach the official mirrors, download the MNIST gzip files separately and drop them into this folder following the standard filenames.

The script auto-detects CUDA and will use GPU if available.

### Corrupted-label experiment

```bash
UV_CACHE_DIR=.uv-cache uv run python main_corrupted.py --epochs 5 --corrupt-frac 0.15
```

This variant replaces 15 % of every class with samples drawn from other classes (without duplicating any images). Set `--corrupt-frac 0` to run the exact same code path without corruption for an apples-to-apples comparison. After training it prints macro precision, recall, and ROC-AUC to illustrate how the corruption impacts downstream quality.

### Automated sweeps and plots

```bash
# run main_corrupted.py for fractions 0.00, 0.10, …, 0.50 and save CSV
UV_CACHE_DIR=.uv-cache uv run python sweep_corruption.py --epochs 2 --step 0.1 --output corruption_results.csv

# plot precision/recall/ROC-AUC vs corruption level, producing corruption_metrics.png
UV_CACHE_DIR=.uv-cache uv run python plot_metrics.py --input corruption_results.csv --output corruption_metrics.png
```

`sweep_corruption.py` accepts custom `--fractions` lists if you want specific values (e.g., `--fractions 0 0.05 0.15 0.5`). `plot_metrics.py` expects the CSV schema emitted by the sweep script and stacks three subplots (precision/recall/ROC-AUC) into one image.
