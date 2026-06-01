import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

from data_loader import build_dataloaders, random_augment
from models import build_model, count_parameters
from dataset_utils import NUM_CLASSES


# ------------------------------------------------------------------
# Trainer
# ------------------------------------------------------------------

class Trainer:
    """
    Wraps the full training / evaluation lifecycle.

    Parameters
    ----------
    model        : nn.Module
    train_loader : DataLoader  (yields (features, labels, lengths))
    val_loader   : DataLoader
    device       : torch.device
    lr           : initial learning rate
    weight_decay : AdamW weight decay
    epochs       : total training epochs
    patience     : early-stopping patience
    output_dir   : where checkpoints and logs are saved
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        lr: float        = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int      = 50,
        patience: int    = 10,
        output_dir: str  = "./runs/default",
    ):
        self.model        = model.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.device       = device
        self.epochs       = epochs
        self.patience     = patience
        self.output_dir   = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.optimizer = AdamW(model.parameters(), lr=lr,
                                weight_decay=weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs,
                                            eta_min=lr / 20)

        self.history: dict[str, list] = {
            "train_loss": [], "train_acc": [],
            "val_loss":   [], "val_acc":   [],
        }
        self.best_val_acc  = 0.0
        self.patience_ctr  = 0

    # ------------------------------------------------------------------

    def _run_epoch(self, loader: DataLoader,
                   train: bool = True) -> tuple[float, float]:
        """Run one epoch. Returns (avg_loss, accuracy)."""
        self.model.train(train)
        total_loss = 0.0
        all_preds, all_labels = [], []

        with torch.set_grad_enabled(train):
            for batch in loader:
                features, labels, lengths = batch
                features = features.to(self.device)
                labels   = labels.to(self.device)
                lengths  = lengths.to(self.device)

                logits = self.model(features, lengths)
                loss   = self.criterion(logits, labels)

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                    self.optimizer.step()

                total_loss += loss.item() * len(labels)
                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(loader.dataset)
        accuracy = accuracy_score(all_labels, all_preds)
        return avg_loss, accuracy

    # ------------------------------------------------------------------

    def train(self) -> dict:
        """Run the full training loop with early stopping."""
        print(f"\nTraining on {self.device}  |  "
              f"{count_parameters(self.model):,} parameters\n")
        print(f"{'Epoch':>6}  {'Train Loss':>10}  {'Train Acc':>9}  "
              f"{'Val Loss':>8}  {'Val Acc':>7}  {'LR':>8}")
        print("-" * 65)

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()

            tr_loss, tr_acc = self._run_epoch(self.train_loader, train=True)
            va_loss, va_acc = self._run_epoch(self.val_loader,   train=False)
            self.scheduler.step()

            self.history["train_loss"].append(tr_loss)
            self.history["train_acc"].append(tr_acc)
            self.history["val_loss"].append(va_loss)
            self.history["val_acc"].append(va_acc)

            lr_now = self.optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t0
            print(f"{epoch:>6d}  {tr_loss:>10.4f}  {tr_acc:>9.4f}  "
                  f"{va_loss:>8.4f}  {va_acc:>7.4f}  {lr_now:>8.6f}  "
                  f"({elapsed:.1f}s)")

            # ---- checkpoint ----
            if va_acc > self.best_val_acc:
                self.best_val_acc = va_acc
                self.patience_ctr = 0
                self._save_checkpoint("best_model.pt")
                print(f"          ✓ New best val acc: {va_acc:.4f}")
            else:
                self.patience_ctr += 1
                if self.patience_ctr >= self.patience:
                    print(f"\nEarly stopping at epoch {epoch}.")
                    break

        # Save history
        with open(self.output_dir / "history.json", "w") as f:
            json.dump(self.history, f, indent=2)

        return self.history

    # ------------------------------------------------------------------

    def _save_checkpoint(self, filename: str) -> None:
        torch.save(self.model.state_dict(),
                   self.output_dir / filename)

    def load_best(self) -> None:
        self.model.load_state_dict(
            torch.load(self.output_dir / "best_model.pt",
                       map_location=self.device)
        )

    # ------------------------------------------------------------------

    def evaluate(self, loader: DataLoader,
                 split_name: str = "Test") -> dict:
        """
        Full evaluation on any split. Returns accuracy, macro F1.
        """
        self.model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for features, labels, lengths in loader:
                features = features.to(self.device)
                lengths  = lengths.to(self.device)
                logits   = self.model(features, lengths)
                preds    = logits.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())

        acc = accuracy_score(all_labels, all_preds)
        f1  = f1_score(all_labels, all_preds, average="macro", zero_division=0)

        results = {
            "split":     split_name,
            "accuracy":  float(acc),
            "f1_macro":  float(f1),
            "preds":     all_preds,
            "labels":    all_labels,
        }

        print(f"\n{split_name} Results")
        print(f"  Accuracy : {acc:.4f} ({acc*100:.2f} %)")
        print(f"  F1-Score : {f1:.4f}  (macro)")

        return results


# ------------------------------------------------------------------
# CLI entry-point
# ------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a GRU-based environmental sound classifier."
    )
    # Data
    p.add_argument("--dataset_root", type=str, required=True,
                   help="Path to the UrbanSound8K root directory")
    p.add_argument("--feature_type", type=str, default="combined",
                   choices=["mel", "mfcc", "energy", "chroma",
                            "contrast", "zcr", "combined"])
    p.add_argument("--cache_dir", type=str, default=None,
                   help="Directory to cache pre-computed features")
    p.add_argument("--augment", action="store_true",
                   help="Apply on-the-fly audio augmentation during training")

    # Model
    p.add_argument("--model", type=str, default="gru",
                   choices=["gru", "cnn_gru", "attn_gru"])
    p.add_argument("--hidden_size", type=int, default=128)
    p.add_argument("--num_layers",  type=int, default=2)
    p.add_argument("--dropout",     type=float, default=0.3)

    # Training
    p.add_argument("--epochs",      type=int,   default=50)
    p.add_argument("--batch_size",  type=int,   default=64)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--weight_decay",type=float, default=1e-4)
    p.add_argument("--patience",    type=int,   default=10)

    # Misc
    p.add_argument("--output_dir",  type=str,   default="./runs/exp1")
    p.add_argument("--seed",        type=int,   default=42)
    return p.parse_args()


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    augment_fn = random_augment if args.augment else None

    train_loader, val_loader, test_loader, normalizer = build_dataloaders(
        dataset_root=args.dataset_root,
        feature_type=args.feature_type,
        batch_size=args.batch_size,
        cache_dir=args.cache_dir,
        augment=augment_fn,
    )

    # Infer input_size from a single batch
    sample_feat, _, _ = next(iter(train_loader))
    input_size = sample_feat.shape[2]
    print(f"Input feature dim: {input_size}")

    model = build_model(
        args.model,
        input_size=input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        patience=args.patience,
        output_dir=args.output_dir,
    )

    trainer.train()
    trainer.load_best()

    val_results  = trainer.evaluate(val_loader,  split_name="Validation")
    test_results = trainer.evaluate(test_loader, split_name="Test")

    # Save results
    out_path = Path(args.output_dir)
    with open(out_path / "results.json", "w") as f:
        json.dump({
            "args":         vars(args),
            "validation":   {k: v for k, v in val_results.items()
                             if k not in ("preds", "labels")},
            "test":         {k: v for k, v in test_results.items()
                             if k not in ("preds", "labels")},
        }, f, indent=2)

    print(f"\nResults saved to {args.output_dir}/results.json")


if __name__ == "__main__":
    main()
