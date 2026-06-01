import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

from data_loader import build_dataloaders, random_augment
from models import build_model
from train import Trainer, set_seed
from evaluate import (
    plot_confusion_matrix,
    plot_training_history,
    compare_models,
    print_classification_report,
    find_most_confused_pairs,
)


# ------------------------------------------------------------------
# Experiment configuration
# ------------------------------------------------------------------

EXPERIMENTS = [
    # (feature_type, model_name, hidden_size, num_layers, dropout, augment)
    # ----  Feature comparison (GRU baseline)  ----
    ("mel",      "gru", 128, 2, 0.3, False),
    ("mfcc",     "gru", 128, 2, 0.3, False),
    ("combined", "gru", 128, 2, 0.3, False),
    # ----  Hyperparameter sweep  ----
    ("combined", "gru", 256, 2, 0.3, False),
    ("combined", "gru", 128, 3, 0.3, False),
    ("combined", "gru", 128, 2, 0.5, False),
    # ----  Augmentation  ----
    ("combined", "gru", 128, 2, 0.3, True),
    # ----  Bonus architectures  ----
    ("combined", "cnn_gru",  128, 2, 0.3, False),
]


# ------------------------------------------------------------------
# Single experiment runner
# ------------------------------------------------------------------

def run_one(
    dataset_root: str,
    feature_type: str,
    model_name: str,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    augment: bool,
    epochs: int,
    batch_size: int,
    lr: float,
    cache_dir: str | None,
    base_output_dir: str,
    device: torch.device,
    seed: int,
) -> dict:

    tag = (f"{model_name}_{feature_type}_h{hidden_size}"
           f"_l{num_layers}_d{int(dropout*10)}"
           f"{'_aug' if augment else ''}")
    output_dir = Path(base_output_dir) / tag
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  Experiment: {tag}")
    print(f"{'='*70}")

    set_seed(seed)

    augment_fn = random_augment if augment else None

    train_loader, val_loader, test_loader, normalizer = build_dataloaders(
        dataset_root=dataset_root,
        feature_type=feature_type,
        batch_size=batch_size,
        cache_dir=cache_dir,
        augment=augment_fn,
    )

    # Infer input_size
    sample_feat, _, _ = next(iter(train_loader))
    input_size = sample_feat.shape[2]

    model = build_model(
        model_name,
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=lr,
        epochs=epochs,
        patience=max(8, epochs // 5),
        output_dir=str(output_dir),
    )

    history = trainer.train()
    trainer.load_best()

    val_results  = trainer.evaluate(val_loader,  split_name="Validation")
    test_results = trainer.evaluate(test_loader, split_name="Test")

    # ---- Plots ----
    plot_training_history(
        history,
        title=f"Training History — {tag}",
        save_path=str(output_dir / "history.png"),
    )

    plot_confusion_matrix(
        test_results["labels"],
        test_results["preds"],
        title=f"Confusion Matrix — {tag}",
        save_path=str(output_dir / "confusion_matrix.png"),
    )

    print_classification_report(test_results["labels"], test_results["preds"])
    find_most_confused_pairs(test_results["labels"], test_results["preds"])

    # ---- Save results ----
    results = {
        "tag": tag,
        "args": {
            "model": model_name,
            "feature_type": feature_type,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "augment": augment,
        },
        "validation": {
            "accuracy": val_results["accuracy"],
            "f1_macro": val_results["f1_macro"],
        },
        "test": {
            "accuracy": test_results["accuracy"],
            "f1_macro": test_results["f1_macro"],
        },
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


# ------------------------------------------------------------------
# Summary table
# ------------------------------------------------------------------

def print_summary(all_results: list[dict]) -> None:
    print("\n" + "=" * 80)
    print(f"  {'Experiment':<45}  {'Val Acc':>7}  {'Val F1':>6}  "
          f"{'Test Acc':>8}  {'Test F1':>7}")
    print("=" * 80)
    for r in sorted(all_results, key=lambda x: -x["test"]["accuracy"]):
        print(f"  {r['tag']:<45}  "
              f"{r['validation']['accuracy']:>7.4f}  "
              f"{r['validation']['f1_macro']:>6.4f}  "
              f"{r['test']['accuracy']:>8.4f}  "
              f"{r['test']['f1_macro']:>7.4f}")
    print("=" * 80)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_root", type=str, default="data/UrbanSound8K")
    p.add_argument("--epochs",       type=int, default=50)
    p.add_argument("--batch_size",   type=int, default=64)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--cache_dir",    type=str, default="./feature_cache")
    p.add_argument("--output_dir",   type=str, default="./runs")
    p.add_argument("--seed",         type=int, default=42)
    # Run only a subset for quick testing
    p.add_argument("--subset",       type=int, default=None,
                   help="Only run the first N experiments (for debugging)")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    exps = EXPERIMENTS
    if args.subset:
        exps = exps[: args.subset]

    all_results = []
    exp_dirs    = []

    for feat, mdl, hs, nl, do, aug in exps:
        try:
            tag = (f"{mdl}_{feat}_h{hs}_l{nl}_d{int(do*10)}"
                   f"{'_aug' if aug else ''}")
            exp_dir = str(Path(args.output_dir) / tag)
            exp_dirs.append(exp_dir)

            res = run_one(
                dataset_root=args.dataset_root,
                feature_type=feat,
                model_name=mdl,
                hidden_size=hs,
                num_layers=nl,
                dropout=do,
                augment=aug,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                cache_dir=args.cache_dir,
                base_output_dir=args.output_dir,
                device=device,
                seed=args.seed,
            )
            all_results.append(res)

        except Exception as e:
            print(f"[ERROR] Experiment ({feat}, {mdl}) failed: {e}")
            import traceback; traceback.print_exc()

    # ---- Final summary ----
    print_summary(all_results)

    # ---- Overall comparison chart ----
    compare_models(
        exp_dirs,
        split="test",
        save_path=str(Path(args.output_dir) / "model_comparison.png"),
    )

    # Save master results
    with open(Path(args.output_dir) / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nAll done. Results in: {args.output_dir}")


if __name__ == "__main__":
    main()
