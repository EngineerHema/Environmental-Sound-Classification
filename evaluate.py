import json
from pathlib import Path
from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score,
)

from dataset_utils import CLASS_LABELS


# ------------------------------------------------------------------
# Confusion Matrix
# ------------------------------------------------------------------

def plot_confusion_matrix(
    labels: Sequence[int],
    preds:  Sequence[int],
    class_names: list[str] = CLASS_LABELS,
    normalize: bool = True,
    title: str = "Confusion Matrix",
    figsize: tuple = (12, 10),
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plot a styled confusion matrix.

    Parameters
    ----------
    labels      : ground-truth class indices
    preds       : predicted class indices
    normalize   : show row-normalised (recall) percentages
    save_path   : if given, save to this path
    """
    cm = confusion_matrix(labels, preds)
    if normalize:
        cm_plot = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
        fmt, vmax = ".2f", 1.0
    else:
        cm_plot = cm
        fmt, vmax = "d", cm.max()

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        cm_plot,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.4,
        linecolor="#e0e0e0",
        vmin=0,
        vmax=vmax,
        ax=ax,
        annot_kws={"size": 8},
    )

    ax.set_title(title, fontsize=14, pad=14)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45,
                        ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved confusion matrix → {save_path}")

    return fig


# ------------------------------------------------------------------
# Training history
# ------------------------------------------------------------------

def plot_training_history(
    history: dict,
    title: str = "Training History",
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plot train/val loss and accuracy from a history dict.

    Parameters
    ----------
    history  : dict with keys "train_loss", "val_loss", "train_acc", "val_acc"
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle(title, fontsize=13)

    # Loss
    ax1.plot(epochs, history["train_loss"], label="Train", color="#2196F3")
    ax1.plot(epochs, history["val_loss"],   label="Val",   color="#FF5722", linestyle="--")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Cross-Entropy Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Accuracy
    ax2.plot(epochs, history["train_acc"], label="Train", color="#4CAF50")
    ax2.plot(epochs, history["val_acc"],   label="Val",   color="#FF9800", linestyle="--")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy")
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved history plot → {save_path}")

    return fig


# ------------------------------------------------------------------
# Model comparison
# ------------------------------------------------------------------

def compare_models(
    experiment_dirs: list[str],
    split: str = "test",
    save_path: str | None = None,
) -> plt.Figure:
    """
    Bar chart comparing accuracy and macro-F1 across multiple run directories.

    Each run directory must contain results.json (produced by train.py).

    Parameters
    ----------
    experiment_dirs : list of paths to run directories
    split           : "test" or "validation"
    """
    names, accs, f1s = [], [], []

    for d in experiment_dirs:
        p = Path(d) / "results.json"
        if not p.exists():
            print(f"[WARN] No results.json in {d}")
            continue
        with open(p) as f:
            res = json.load(f)
        key = split if split in res else "test"
        names.append(res["args"].get("model", Path(d).name) +
                     f"\n({res['args'].get('feature_type', '')})")
        accs.append(res[key]["accuracy"])
        f1s.append(res[key]["f1_macro"])

    x = np.arange(len(names))
    w = 0.35

    fig, ax = plt.subplots(figsize=(max(8, 2 * len(names)), 5))
    bars1 = ax.bar(x - w / 2, accs, w, label="Accuracy", color="#2196F3", alpha=0.85)
    bars2 = ax.bar(x + w / 2, f1s,  w, label="F1-Macro", color="#FF5722", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(f"Model Comparison  ({split.title()} set)")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    for bar in (*bars1, *bars2):
        ax.annotate(f"{bar.get_height():.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved comparison chart → {save_path}")

    return fig


# ------------------------------------------------------------------
# Per-class report
# ------------------------------------------------------------------

def print_classification_report(
    labels: Sequence[int],
    preds:  Sequence[int],
    class_names: list[str] = CLASS_LABELS,
) -> None:
    """Print a scikit-learn classification report."""
    print("\nClassification Report")
    print("=" * 65)
    print(classification_report(
        labels, preds, target_names=class_names, digits=4, zero_division=0
    ))


# ------------------------------------------------------------------
# Most confused pairs
# ------------------------------------------------------------------

def find_most_confused_pairs(
    labels: Sequence[int],
    preds:  Sequence[int],
    class_names: list[str] = CLASS_LABELS,
    top_k: int = 5,
) -> list[dict]:
    """
    Return the top-k most confused class pairs (off-diagonal CM entries).

    Returns
    -------
    list of dicts: {"true", "predicted", "count", "recall_loss"}
    """
    cm = confusion_matrix(labels, preds)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = cm.astype(float) / (row_sums + 1e-8)

    # Zero the diagonal
    np.fill_diagonal(cm_norm, 0)

    # Flatten and sort
    flat = cm_norm.flatten()
    top_idx = np.argsort(flat)[::-1][:top_k]

    pairs = []
    for idx in top_idx:
        r, c = divmod(int(idx), len(class_names))
        pairs.append({
            "true":        class_names[r],
            "predicted":   class_names[c],
            "count":       int(cm[r, c]),
            "recall_loss": float(cm_norm[r, c]),
        })

    print(f"\nTop-{top_k} Most Confused Pairs")
    print("-" * 55)
    for p in pairs:
        print(f"  {p['true']:<22s} → {p['predicted']:<22s} "
              f"  {p['count']:4d} samples  ({p['recall_loss']:.2%})")

    return pairs


# ------------------------------------------------------------------
# Quick evaluation from saved checkpoint
# ------------------------------------------------------------------

def evaluate_from_checkpoint(
    checkpoint_path: str,
    model_name: str,
    input_size: int,
    test_loader,
    device: str = "cpu",
    output_dir: str = ".",
    **model_kwargs,
) -> dict:
    """
    Load a checkpoint and evaluate on test_loader.
    Saves confusion matrix and prints report.
    """
    import torch
    from models import build_model

    model = build_model(model_name, input_size=input_size, **model_kwargs)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval().to(device)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for features, labels, lengths in test_loader:
            features = features.to(device)
            lengths  = lengths.to(device)
            logits   = model(features, lengths)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    print_classification_report(all_labels, all_preds)
    find_most_confused_pairs(all_labels, all_preds)

    cm_fig = plot_confusion_matrix(
        all_labels, all_preds,
        title=f"{model_name} — Test Confusion Matrix",
        save_path=str(Path(output_dir) / "confusion_matrix.png"),
    )

    return {"accuracy": acc, "f1_macro": f1,
            "preds": all_preds, "labels": all_labels}
