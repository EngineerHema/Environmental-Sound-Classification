import os
import pickle
import hashlib
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from dataset_utils import (
    load_audio, load_metadata,
    TRAIN_FOLDS, VAL_FOLDS, TEST_FOLDS,
    SAMPLE_RATE, NUM_CLASSES,
)
from features import extract_features, FeatureNormalizer, FeatureType


# ------------------------------------------------------------------
# UrbanSound8K Dataset
# ------------------------------------------------------------------

class UrbanSound8KDataset(Dataset):
    """
    PyTorch Dataset for UrbanSound8K.

    Parameters
    ----------
    df           : metadata DataFrame (from load_metadata), filtered to the
                   desired folds BEFORE passing in
    feature_type : feature to extract  (see features.py)
    sr           : sample rate
    hop_length   : STFT hop length
    n_fft        : FFT window size
    n_mels       : number of Mel bands
    n_mfcc       : number of MFCC coefficients
    normalizer   : fitted FeatureNormalizer (or None to skip normalisation)
    cache_dir    : if given, pre-computed features are cached here
    augment      : optional callable(waveform) → waveform for on-the-fly aug.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_type: FeatureType = "combined",
        sr: int = SAMPLE_RATE,
        hop_length: int = 512,
        n_fft: int = 1024,
        n_mels: int = 64,
        n_mfcc: int = 40,
        normalizer: Optional[FeatureNormalizer] = None,
        cache_dir: Optional[str] = None,
        augment: Optional[Callable] = None,
    ):
        self.df           = df.reset_index(drop=True)
        self.feature_type = feature_type
        self.sr           = sr
        self.hop_length   = hop_length
        self.n_fft        = n_fft
        self.n_mels       = n_mels
        self.n_mfcc       = n_mfcc
        self.normalizer   = normalizer
        self.cache_dir    = Path(cache_dir) if cache_dir else None
        self.augment      = augment

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row       = self.df.iloc[idx]
        file_path = row["file_path"]
        label     = int(row["classID"])

        # ---- try cache ----
        feat = self._load_from_cache(file_path)

        if feat is None:
            waveform, _ = load_audio(file_path, sr=self.sr)
            if self.augment:
                waveform = self.augment(waveform)
            feat = extract_features(
                waveform,
                sr=self.sr,
                feature_type=self.feature_type,
                hop_length=self.hop_length,
                n_fft=self.n_fft,
                n_mels=self.n_mels,
                n_mfcc=self.n_mfcc,
            )                              # shape: (F, T)
            self._save_to_cache(file_path, feat)

        # transpose to (T, F) for sequence models
        feat = feat.T                      # (T, F)

        if self.normalizer:
            # normalizer expects (N, T, F); add/remove batch dim
            feat = self.normalizer.transform(feat[np.newaxis])[0]

        return torch.tensor(feat, dtype=torch.float32), label

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _cache_key(self, file_path: str) -> str:
        tag = f"{file_path}_{self.feature_type}_{self.sr}_{self.hop_length}"
        return hashlib.md5(tag.encode()).hexdigest() + ".pkl"

    def _load_from_cache(self, file_path: str) -> Optional[np.ndarray]:
        if self.cache_dir is None:
            return None
        p = self.cache_dir / self._cache_key(file_path)
        if p.exists():
            with open(p, "rb") as f:
                return pickle.load(f)
        return None

    def _save_to_cache(self, file_path: str, feat: np.ndarray) -> None:
        if self.cache_dir is None:
            return
        p = self.cache_dir / self._cache_key(file_path)
        with open(p, "wb") as f:
            pickle.dump(feat, f)


# ------------------------------------------------------------------
# Collate: pad/truncate sequences to a fixed length
# ------------------------------------------------------------------

def collate_pad(batch: list, max_len: Optional[int] = None):
    """
    Pad (or truncate) variable-length sequences in a batch to the same length.

    Parameters
    ----------
    batch   : list of (feature_tensor, label) pairs
    max_len : if given, sequences longer than max_len are truncated

    Returns
    -------
    features : (B, T_max, F)
    labels   : (B,)
    lengths  : (B,)   original (unpadded) lengths, useful for pack_padded
    """
    features, labels = zip(*batch)
    lengths  = [f.shape[0] for f in features]
    T_max    = max(lengths) if max_len is None else min(max(lengths), max_len)
    F        = features[0].shape[1]

    padded = torch.zeros(len(features), T_max, F)
    for i, (feat, length) in enumerate(zip(features, lengths)):
        t = min(length, T_max)
        padded[i, :t] = feat[:t]

    return (
        padded,
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(lengths, dtype=torch.long).clamp(max=T_max),
    )


# ------------------------------------------------------------------
# Picklable collate wrapper
# Replaces the lambda so multiprocessing (spawn) can pickle it on
# Python 3.14+ / Windows where lambdas inside functions are not picklable.
# ------------------------------------------------------------------

class CollatePad:
    """Picklable wrapper around collate_pad with a fixed max_len."""

    def __init__(self, max_len: Optional[int] = None):
        self.max_len = max_len

    def __call__(self, batch: list):
        return collate_pad(batch, max_len=self.max_len)


# ------------------------------------------------------------------
# Convenience factory
# ------------------------------------------------------------------

def build_dataloaders(
    dataset_root: str,
    feature_type: FeatureType = "combined",
    batch_size: int = 64,
    num_workers: int = 4,
    cache_dir: Optional[str] = None,
    max_len: Optional[int] = None,
    sr: int = SAMPLE_RATE,
    hop_length: int = 512,
    n_fft: int = 1024,
    n_mels: int = 64,
    n_mfcc: int = 40,
    augment: Optional[Callable] = None,
) -> tuple[DataLoader, DataLoader, DataLoader, FeatureNormalizer]:
    """
    Build train / val / test DataLoaders plus a fitted FeatureNormalizer.

    Returns
    -------
    (train_loader, val_loader, test_loader, normalizer)
    """
    df = load_metadata(dataset_root)

    df_train = df[df["fold"].isin(TRAIN_FOLDS)]
    df_val   = df[df["fold"].isin(VAL_FOLDS)]
    df_test  = df[df["fold"].isin(TEST_FOLDS)]

    kwargs = dict(
        feature_type=feature_type,
        sr=sr,
        hop_length=hop_length,
        n_fft=n_fft,
        n_mels=n_mels,
        n_mfcc=n_mfcc,
        cache_dir=cache_dir,
    )

    # ------ fit normalizer on train split ------
    print("Fitting normalizer on training set (may take a few minutes)…")
    tmp_ds = UrbanSound8KDataset(df_train, augment=None, **kwargs)

    # Collect a random subset (or all) to fit the normalizer
    sample_idx = np.random.choice(len(tmp_ds),
                                  size=min(2000, len(tmp_ds)), replace=False)
    feats = []
    for i in sample_idx:
        feat, _ = tmp_ds[i]          # (T, F)
        feats.append(feat.numpy())
    X_sample = np.stack(feats)       # (N, T, F)  — may vary in T but that's ok
    normalizer = FeatureNormalizer().fit(X_sample)
    print("Normalizer fitted.")

    # ------ build datasets ------
    # Use CollatePad (module-level class) instead of a lambda so that
    # multiprocessing workers can pickle it on Python 3.14+ / Windows.
    collate = CollatePad(max_len=max_len)

    # Only pin memory when a CUDA GPU is actually available.
    pin_memory = torch.cuda.is_available()

    train_ds = UrbanSound8KDataset(df_train, normalizer=normalizer,
                                    augment=augment, **kwargs)
    val_ds   = UrbanSound8KDataset(df_val,   normalizer=normalizer, **kwargs)
    test_ds  = UrbanSound8KDataset(df_test,  normalizer=normalizer, **kwargs)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, collate_fn=collate,
                               pin_memory=pin_memory)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                               num_workers=num_workers, collate_fn=collate,
                               pin_memory=pin_memory)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                               num_workers=num_workers, collate_fn=collate,
                               pin_memory=pin_memory)

    print(f"Train: {len(train_ds):5d}  |  Val: {len(val_ds):5d}  |  Test: {len(test_ds):5d}")
    return train_loader, val_loader, test_loader, normalizer


# ------------------------------------------------------------------
# Simple on-the-fly augmentation helpers
# ------------------------------------------------------------------

def random_augment(waveform: np.ndarray,
                   sr: int = SAMPLE_RATE,
                   noise_std: float = 0.005,
                   pitch_shift_range: int = 2,
                   time_stretch_range: tuple = (0.9, 1.1)) -> np.ndarray:
    """
    Light augmentation pipeline applied per waveform during training.

    Randomly applies:
    - Additive Gaussian noise
    - Time stretching
    - Pitch shifting
    """
    import librosa  # imported here to keep it optional at module level

    choice = np.random.randint(3)

    if choice == 0:
        # Additive noise
        noise = np.random.normal(0, noise_std, waveform.shape).astype(np.float32)
        waveform = waveform + noise

    elif choice == 1:
        # Time stretch
        rate = np.random.uniform(*time_stretch_range)
        waveform = librosa.effects.time_stretch(waveform, rate=rate)

    else:
        # Pitch shift
        steps = np.random.randint(-pitch_shift_range, pitch_shift_range + 1)
        if steps != 0:
            waveform = librosa.effects.pitch_shift(waveform, sr=sr, n_steps=steps)

    # Re-pad / trim to fixed length
    from dataset_utils import N_SAMPLES
    if len(waveform) < N_SAMPLES:
        waveform = np.pad(waveform, (0, N_SAMPLES - len(waveform)))
    return waveform[:N_SAMPLES]