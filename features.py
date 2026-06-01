import numpy as np
import librosa
from typing import Literal

# ------------------------------------------------------------------
# Default hyper-parameters
# ------------------------------------------------------------------
SR          = 22050
N_FFT       = 1024
HOP_LENGTH  = 512
N_MELS      = 64
N_MFCC      = 40
N_CHROMA    = 12
N_CONTRAST  = 6          # librosa returns n_bands+1 = 7 bands

FeatureType = Literal[
    "mel", "mfcc", "energy", "chroma", "contrast", "zcr", "combined"
]


# ------------------------------------------------------------------
# Individual feature extractors
# ------------------------------------------------------------------

def extract_mel(waveform: np.ndarray,
                sr: int = SR,
                n_mels: int = N_MELS,
                n_fft: int = N_FFT,
                hop_length: int = HOP_LENGTH) -> np.ndarray:
    """
    Log-power Mel spectrogram.

    Returns
    -------
    (n_mels, T) float32
    """
    mel = librosa.feature.melspectrogram(
        y=waveform, sr=sr, n_fft=n_fft,
        hop_length=hop_length, n_mels=n_mels
    )
    return librosa.power_to_db(mel, ref=np.max).astype(np.float32)


def extract_mfcc(waveform: np.ndarray,
                 sr: int = SR,
                 n_mfcc: int = N_MFCC,
                 n_fft: int = N_FFT,
                 hop_length: int = HOP_LENGTH) -> np.ndarray:
    """
    MFCCs  + delta  + delta-delta  (3 × n_mfcc features).

    Returns
    -------
    (3*n_mfcc, T) float32
    """
    mfcc  = librosa.feature.mfcc(
        y=waveform, sr=sr, n_mfcc=n_mfcc,
        n_fft=n_fft, hop_length=hop_length
    )
    delta  = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    return np.vstack([mfcc, delta, delta2]).astype(np.float32)


def extract_energy(waveform: np.ndarray,
                   hop_length: int = HOP_LENGTH,
                   n_fft: int = N_FFT) -> np.ndarray:
    """
    Short-time energy (RMS energy per frame).

    Returns
    -------
    (1, T) float32
    """
    rms = librosa.feature.rms(y=waveform, frame_length=n_fft,
                               hop_length=hop_length)
    return rms.astype(np.float32)            # shape (1, T)


def extract_chroma(waveform: np.ndarray,
                   sr: int = SR,
                   n_chroma: int = N_CHROMA,
                   hop_length: int = HOP_LENGTH) -> np.ndarray:
    """
    Chroma (12 pitch classes).

    Returns
    -------
    (12, T) float32
    """
    chroma = librosa.feature.chroma_stft(
        y=waveform, sr=sr, n_chroma=n_chroma, hop_length=hop_length
    )
    return chroma.astype(np.float32)


def extract_spectral_contrast(waveform: np.ndarray,
                               sr: int = SR,
                               n_bands: int = N_CONTRAST,
                               hop_length: int = HOP_LENGTH) -> np.ndarray:
    """
    Spectral contrast — captures the difference between peaks and valleys in a
    spectrum; useful for distinguishing tonal vs. noise-like sounds.

    Returns
    -------
    (n_bands+1, T) float32
    """
    contrast = librosa.feature.spectral_contrast(
        y=waveform, sr=sr, n_bands=n_bands, hop_length=hop_length
    )
    return contrast.astype(np.float32)


def extract_zcr(waveform: np.ndarray,
                hop_length: int = HOP_LENGTH,
                n_fft: int = N_FFT) -> np.ndarray:
    """
    Zero-Crossing Rate — high for noisy/unvoiced sounds (e.g. gun shot).

    Returns
    -------
    (1, T) float32
    """
    zcr = librosa.feature.zero_crossing_rate(
        y=waveform, frame_length=n_fft, hop_length=hop_length
    )
    return zcr.astype(np.float32)


# ------------------------------------------------------------------
# Combined extractor
# ------------------------------------------------------------------

def extract_features(waveform: np.ndarray,
                     sr: int = SR,
                     feature_type: FeatureType = "combined",
                     hop_length: int = HOP_LENGTH,
                     n_fft: int = N_FFT,
                     n_mels: int = N_MELS,
                     n_mfcc: int = N_MFCC) -> np.ndarray:
    """
    Extract the requested feature representation.

    Parameters
    ----------
    waveform     : 1-D float32 waveform at `sr` Hz
    sr           : sample rate
    feature_type : one of "mel" | "mfcc" | "energy" | "chroma" |
                   "contrast" | "zcr" | "combined"
    hop_length   : STFT hop length (frames)
    n_fft        : FFT window size
    n_mels       : number of Mel bands
    n_mfcc       : number of MFCC coefficients

    Returns
    -------
    ndarray of shape (n_features, T) — float32
    """
    extractors = {
        "mel":      lambda: extract_mel(waveform, sr, n_mels, n_fft, hop_length),
        "mfcc":     lambda: extract_mfcc(waveform, sr, n_mfcc, n_fft, hop_length),
        "energy":   lambda: extract_energy(waveform, hop_length, n_fft),
        "chroma":   lambda: extract_chroma(waveform, sr, N_CHROMA, hop_length),
        "contrast": lambda: extract_spectral_contrast(waveform, sr, N_CONTRAST, hop_length),
        "zcr":      lambda: extract_zcr(waveform, hop_length, n_fft),
    }

    if feature_type in extractors:
        return extractors[feature_type]()

    if feature_type == "combined":
        parts = [v() for v in extractors.values()]
        # All parts have shape (F_i, T); concatenate along feature axis
        T = min(p.shape[1] for p in parts)
        return np.concatenate([p[:, :T] for p in parts], axis=0)

    raise ValueError(
        f"Unknown feature_type='{feature_type}'. "
        f"Choose from: {list(extractors.keys()) + ['combined']}"
    )


# ------------------------------------------------------------------
# Normalisation (per-feature z-score, computed on training set)
# ------------------------------------------------------------------

class FeatureNormalizer:
    """
    Fit mean/std on training data, then normalise any split.

    Usage
    -----
    >>> norm = FeatureNormalizer()
    >>> norm.fit(X_train)       # X_train: (N, T, F)
    >>> X_train = norm.transform(X_train)
    >>> X_val   = norm.transform(X_val)
    """

    def __init__(self):
        self.mean_: np.ndarray | None = None
        self.std_:  np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "FeatureNormalizer":
        """X shape: (N, T, F) or (N, F)."""
        self.mean_ = X.mean(axis=(0, 1) if X.ndim == 3 else 0, keepdims=True)
        self.std_  = X.std( axis=(0, 1) if X.ndim == 3 else 0, keepdims=True) + 1e-8
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.mean_ is not None, "Call .fit() first."
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def save(self, path: str) -> None:
        np.savez(path, mean=self.mean_, std=self.std_)

    def load(self, path: str) -> "FeatureNormalizer":
        data = np.load(path)
        self.mean_ = data["mean"]
        self.std_  = data["std"]
        return self


# ------------------------------------------------------------------
# Feature dimension reference
# ------------------------------------------------------------------

def feature_dim(feature_type: FeatureType,
                n_mels: int = N_MELS,
                n_mfcc: int = N_MFCC) -> int:
    """Return the number of feature channels for a given feature type."""
    dims = {
        "mel":      n_mels,
        "mfcc":     3 * n_mfcc,
        "energy":   1,
        "chroma":   N_CHROMA,
        "contrast": N_CONTRAST + 1,
        "zcr":      1,
    }
    if feature_type in dims:
        return dims[feature_type]
    if feature_type == "combined":
        return sum(dims.values())
    raise ValueError(f"Unknown feature_type='{feature_type}'")


if __name__ == "__main__":
    # Quick smoke-test
    dummy = np.random.randn(22050 * 4).astype(np.float32)
    for ft in ["mel", "mfcc", "energy", "chroma", "contrast", "zcr", "combined"]:
        feat = extract_features(dummy, feature_type=ft)
        print(f"{ft:12s}  shape: {feat.shape}  "
              f"expected_F={feature_dim(ft)}")
