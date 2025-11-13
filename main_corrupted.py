import argparse
import math
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

NUM_CLASSES = 10


class SimpleMLP(nn.Module):
    """A tiny fully-connected network that works well on MNIST."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def corrupt_training_set(
    dataset: datasets.MNIST, fraction: float, seed: int = 0
) -> datasets.MNIST:
    """Relabel a fraction of each class with images pulled from other classes."""

    if fraction < 0 or fraction > 0.9:
        raise ValueError("fraction must be in [0, 0.9] for a meaningful corruption sweep.")
    if fraction == 0:
        return dataset

    rng = random.Random(seed)
    targets = dataset.targets.clone()
    class_indices: Dict[int, List[int]] = {}
    donors_by_label: Dict[int, List[int]] = {}
    n_corrupt: Dict[int, int] = {}

    for label in range(NUM_CLASSES):
        idxs = torch.where(targets == label)[0].tolist()
        rng.shuffle(idxs)
        corrupt_count = max(1, int(len(idxs) * fraction))
        donors_by_label[label] = idxs[:corrupt_count]
        class_indices[label] = idxs[corrupt_count:]
        n_corrupt[label] = corrupt_count

    donor_entries: List[Tuple[int, int]] = []
    for label, idxs in donors_by_label.items():
        for idx in idxs:
            donor_entries.append((idx, label))

    recipients_template: List[int] = []
    for label, count in n_corrupt.items():
        recipients_template.extend([label] * count)

    assignments = build_corruption_assignment(donor_entries, recipients_template, rng)

    idx_list = list(assignments.keys())
    new_labels = torch.tensor([assignments[idx] for idx in idx_list], dtype=targets.dtype)
    targets[idx_list] = new_labels
    dataset.targets = targets
    return dataset


def build_corruption_assignment(
    donor_entries: List[Tuple[int, int]], recipient_template: List[int], rng: random.Random
) -> Dict[int, int]:
    """Assign donor samples to new class labels while avoiding duplicates."""

    donors = donor_entries[:]
    rng.shuffle(donors)

    for _ in range(50):
        recipients = recipient_template[:]
        rng.shuffle(recipients)
        if resolve_conflicts(donors, recipients):
            return {idx: new_label for (idx, _), new_label in zip(donors, recipients)}
    raise RuntimeError("Failed to assign corrupted samples without collisions.")


def resolve_conflicts(donors: List[Tuple[int, int]], recipients: List[int]) -> bool:
    """Swap recipient labels until no donor keeps its original label."""

    for i, ((_, donor_label), recipient_label) in enumerate(zip(donors, recipients)):
        if donor_label == recipient_label:
            swap_found = False
            for j in range(i + 1, len(recipients)):
                other_donor_label = donors[j][1]
                if donor_label != recipients[j] and other_donor_label != recipient_label:
                    recipients[i], recipients[j] = recipients[j], recipients[i]
                    swap_found = True
                    break
            if not swap_found:
                return False
    return True


def build_loaders(batch_size: int, data_dir: str, corrupt_frac: float, seed: int):
    """Create loaders with a corrupted training split."""

    transform = transforms.ToTensor()
    train_full = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    corrupt_training_set(train_full, corrupt_frac, seed)

    val_size = 10_000
    train_size = len(train_full) - val_size
    train_ds, val_ds = random_split(
        train_full,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    loader_kwargs = {"batch_size": batch_size, "num_workers": 0, "pin_memory": torch.cuda.is_available()}
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader


def run_epoch(loader, model, loss_fn, device, optimizer=None):
    """Runs train or eval epoch depending on optimizer presence."""

    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        if is_train:
            optimizer.zero_grad()
            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                logits = model(images)
                loss = loss_fn(logits, labels)

        total_loss += loss.item() * images.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_examples += images.size(0)

    avg_loss = total_loss / total_examples
    avg_acc = total_correct / total_examples
    return avg_loss, avg_acc


def evaluate_with_metrics(loader, model, loss_fn, device):
    """Compute loss/acc plus macro precision, recall, and ROC AUC on a loader."""

    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    preds, truths, probs = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = loss_fn(logits, labels)

            total_loss += loss.item() * images.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_examples += images.size(0)

            preds.append(logits.argmax(dim=1).cpu())
            truths.append(labels.cpu())
            probs.append(torch.softmax(logits, dim=1).cpu())

    y_true = torch.cat(truths).numpy()
    y_pred = torch.cat(preds).numpy()
    y_prob = torch.cat(probs).numpy()

    precision = macro_precision(y_true, y_pred)
    recall = macro_recall(y_true, y_pred)
    roc_auc = macro_roc_auc(y_true, y_prob)

    avg_loss = total_loss / total_examples
    avg_acc = total_correct / total_examples
    return avg_loss, avg_acc, precision, recall, roc_auc


def parse_args():
    parser = argparse.ArgumentParser(
        description="MNIST classifier with a corrupted training distribution."
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--data-dir", type=str, default="data", help="Where to cache/download MNIST.")
    parser.add_argument(
        "--corrupt-frac",
        type=float,
        default=0.15,
        help="Fraction of each class to replace with samples from other classes.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device} with {args.corrupt_frac:.0%} corruption")

    train_loader, val_loader, test_loader = build_loaders(
        args.batch_size, args.data_dir, args.corrupt_frac, args.seed
    )
    model = SimpleMLP().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(train_loader, model, loss_fn, device, optimizer=optimizer)
        val_loss, val_acc = run_epoch(val_loader, model, loss_fn, device)
        print(
            f"Epoch {epoch:02d} | "
            f"train loss {train_loss:.3f}, acc {train_acc*100:5.2f}% | "
            f"val loss {val_loss:.3f}, acc {val_acc*100:5.2f}%"
        )

    test_loss, test_acc, precision, recall, roc_auc = evaluate_with_metrics(
        test_loader, model, loss_fn, device
    )
    print(
        f"Test loss {test_loss:.3f}, acc {test_acc*100:5.2f}% | "
        f"precision {precision:.3f} | recall {recall:.3f} | ROC AUC {roc_auc:.3f}"
    )


def macro_precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    cm = confusion_matrix(y_true, y_pred)
    precision_per_class = safe_divide(np.diag(cm), cm.sum(axis=0))
    return precision_per_class.mean()


def macro_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    cm = confusion_matrix(y_true, y_pred)
    recall_per_class = safe_divide(np.diag(cm), cm.sum(axis=1))
    return recall_per_class.mean()


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    result = np.zeros_like(numerator, dtype=np.float64)
    mask = denominator != 0
    result[mask] = numerator[mask] / denominator[mask]
    return result


def macro_roc_auc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    aucs = []
    for cls in range(NUM_CLASSES):
        binary_true = (y_true == cls).astype(np.int32)
        aucs.append(binary_auc(binary_true, y_scores[:, cls]))
    finite_aucs = [auc for auc in aucs if not math.isnan(auc)]
    return float(np.mean(finite_aucs)) if finite_aucs else float("nan")


def binary_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    pos = y_true.sum()
    neg = len(y_true) - pos
    if pos == 0 or neg == 0:
        return math.nan
    order = np.argsort(scores)[::-1]
    y_true_sorted = y_true[order]
    tps = np.cumsum(y_true_sorted)
    fps = np.cumsum(1 - y_true_sorted)
    tpr = tps / pos
    fpr = fps / neg
    return float(np.trapezoid(tpr, fpr))


if __name__ == "__main__":
    main()
