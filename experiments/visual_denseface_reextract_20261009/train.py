#!/usr/bin/env python3
"""Train the DenseFace-compatible backbone on hard-label FER+."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ferplus import EMOTIONS, FerPlusDataset, compute_train_mean_std  # noqa: E402
from model import DenseFace  # noqa: E402


DEFAULT_EARLY_STOPPING_PATIENCE = 8


def checkpoint_selection_score(metrics: dict) -> float:
    """MEmoBert-compatible checkpoint criterion: development macro-F1."""
    return float(metrics["macro_f1"])


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def classification_metrics(targets: list[int], predictions: list[int], classes: int) -> dict:
    confusion = np.zeros((classes, classes), dtype=np.int64)
    for target, prediction in zip(targets, predictions):
        confusion[target, prediction] += 1
    support = confusion.sum(axis=1)
    f1 = []
    for index in range(classes):
        tp = confusion[index, index]
        fp = confusion[:, index].sum() - tp
        fn = confusion[index, :].sum() - tp
        denominator = 2 * tp + fp + fn
        f1.append(float(2 * tp / denominator) if denominator else 0.0)
    total = int(support.sum())
    return {
        "accuracy": float(np.trace(confusion) / total) if total else 0.0,
        "macro_f1": float(np.mean(f1)),
        "weighted_f1": float(np.average(f1, weights=support)) if total else 0.0,
        "per_class_f1": dict(zip(EMOTIONS, f1)),
        "support": dict(zip(EMOTIONS, map(int, support))),
        "confusion": confusion.tolist(),
    }


def run_epoch(model, loader, criterion, device, optimizer=None) -> dict:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    targets: list[int] = []
    predictions: list[int] = []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
        total_loss += float(loss.detach()) * labels.numel()
        targets.extend(labels.cpu().tolist())
        predictions.extend(logits.argmax(dim=1).cpu().tolist())
    metrics = classification_metrics(targets, predictions, len(EMOTIONS))
    metrics["loss"] = total_loss / max(len(targets), 1)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fer2013-csv", type=Path, required=True)
    parser.add_argument("--fer2013new-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--optimizer", choices=("adam", "sgd"), default="adam")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--patience", type=int, default=DEFAULT_EARLY_STOPPING_PATIENCE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--smoke", action="store_true", help="run one batch for plumbing validation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    train_mean, train_std, normalization_images = compute_train_mean_std(
        args.fer2013_csv, args.fer2013new_csv
    )
    datasets = {
        split: FerPlusDataset(
            args.fer2013_csv,
            args.fer2013new_csv,
            split,
            augment=split == "train" and not args.smoke,
            mean=train_mean,
            std=train_std,
        )
        for split in ("train", "dev", "test")
    }
    generator = torch.Generator().manual_seed(args.seed)
    loader_datasets = {
        name: Subset(dataset, range(min(2, len(dataset)))) if args.smoke else dataset
        for name, dataset in datasets.items()
    }
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=min(args.batch_size, 2) if args.smoke else args.batch_size,
            shuffle=split == "train",
            num_workers=0 if args.smoke else args.workers,
            pin_memory=device.type == "cuda",
            generator=generator if split == "train" else None,
        )
        for split, dataset in loader_datasets.items()
    }
    model = DenseFace(num_classes=len(EMOTIONS)).to(device)
    criterion = nn.CrossEntropyLoss()
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.learning_rate,
            momentum=args.momentum,
            nesterov=True,
            weight_decay=args.weight_decay,
        )
    epochs = 1 if args.smoke else args.epochs
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[max(1, epochs // 2), max(2, 3 * epochs // 4)],
        gamma=0.1,
    )
    history = []
    if args.patience < 1:
        raise ValueError("--patience must be at least 1")
    best_dev_macro_f1 = -1.0
    epochs_without_improvement = 0
    checkpoint_path = args.output_dir / "best_dev.pt"
    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(model, loaders["train"], criterion, device, optimizer)
        dev_metrics = run_epoch(model, loaders["dev"], criterion, device)
        entry = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "dev": dev_metrics,
        }
        history.append(entry)
        print(json.dumps(entry, sort_keys=True))
        selection_score = checkpoint_selection_score(dev_metrics)
        if selection_score > best_dev_macro_f1:
            best_dev_macro_f1 = selection_score
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model": {
                        "name": "DenseNet-BC-100",
                        "growth_rate": 12,
                        "block_layers": [16, 16, 16],
                        "compression": 0.5,
                        "feature_dim": model.feature_dim,
                        "num_classes": len(EMOTIONS),
                    },
                    "epoch": epoch,
                    "dev": dev_metrics,
                    "selection": {"criterion": "dev_macro_f1", "score": selection_score},
                    "seed": args.seed,
                    "optimizer": {
                        "name": args.optimizer,
                        "learning_rate": args.learning_rate,
                        "weight_decay": args.weight_decay,
                        "momentum": args.momentum if args.optimizer == "sgd" else None,
                    },
                    "normalization": {
                        "source": "kept FER+ Training images",
                        "mean": train_mean,
                        "std": train_std,
                        "images": normalization_images,
                    },
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
        scheduler.step()
        if epochs_without_improvement >= args.patience:
            print(
                f"early stopping after {args.patience} epochs without "
                "development macro-F1 improvement"
            )
            break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test_metrics = run_epoch(model, loaders["test"], criterion, device)
    (args.output_dir / "metrics.json").write_text(
        json.dumps({"history": history, "best_dev": checkpoint["dev"], "test": test_metrics}, indent=2),
        encoding="utf-8",
    )
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = None
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "architecture": "DenseNet-BC-100 (growth=12, blocks=16/16/16, compression=0.5)",
        "feature_dim": 342,
        "label_policy": "argmax over 10 FER+ votes; drop samples won by unknown/NF or all-zero votes",
        "split_policy": USAGE_DESCRIPTION,
        "normalization": {
            "source": "kept FER+ Training images",
            "mean": train_mean,
            "std": train_std,
            "images": normalization_images,
        },
        "seed": args.seed,
        "optimizer": {
            "name": args.optimizer,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "momentum": args.momentum if args.optimizer == "sgd" else None,
            "scheduler": "MultiStepLR(0.5*epochs, 0.75*epochs; gamma=0.1)",
        },
        "checkpoint_selection": {
            "criterion": "dev_macro_f1",
            "early_stopping_patience": args.patience,
            "best_score": checkpoint["selection"]["score"],
            "best_epoch": checkpoint["epoch"],
        },
        "smoke": args.smoke,
        "dataset_sizes": {name: len(dataset) for name, dataset in datasets.items()},
        "filter_stats": datasets["train"].filter_stats,
        "inputs": {
            str(args.fer2013_csv.resolve()): sha256(args.fer2013_csv),
            str(args.fer2013new_csv.resolve()): sha256(args.fer2013new_csv),
        },
        "checkpoint": {"path": str(checkpoint_path), "sha256": sha256(checkpoint_path)},
        "git_commit": git_commit,
        "torch": torch.__version__,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


USAGE_DESCRIPTION = {"Training": "train", "PublicTest": "dev", "PrivateTest": "test"}


if __name__ == "__main__":
    main()
