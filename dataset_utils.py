import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import IPython.display as ipd

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
CLASS_LABELS = [
    "air_conditioner", "car_horn", "children_playing", "dog_bark",
    "drilling", "engine_idling", "gun_shot", "jackhammer",
    "siren", "street_music",
]
NUM_CLASSES   = len(CLASS_LABELS)
SAMPLE_RATE   = 22050   # Hz  – librosa default
CLIP_DURATION = 4.0     # seconds (max clip length in dataset)
N_SAMPLES     = int(SAMPLE_RATE * CLIP_DURATION)

# Fold splits as required by the assignment
TRAIN_FOLDS = [1, 2, 3, 4, 5, 6]
VAL_FOLDS   = [7, 8]
TEST_FOLDS  = [9, 10]


# ------------------------------------------------------------------
# Loading
# ------------------------------------------------------------------

def load_audio(file_path: str,
               sr: int = SAMPLE_RATE,
               duration: float = CLIP_DURATION,
               mono: bool = True) -> tuple[np.ndarray, int]:

    waveform, sr_orig = librosa.load(file_path, sr=sr, mono=mono,
                                     duration=duration)
    # Zero-pad if the clip is shorter than `duration`
    target_len = int(sr * duration)
    if len(waveform) < target_len:
        waveform = np.pad(waveform, (0, target_len - len(waveform)))
    else:
        waveform = waveform[:target_len]
    return waveform, sr


def load_metadata(dataset_root: str) -> pd.DataFrame:
    """
    Load UrbanSound8K.csv and add a column with the absolute audio path.

    Parameters
    ----------
    dataset_root : top-level directory that contains 'audio/' and 'metadata/'

    Returns
    -------
    DataFrame with columns including 'file_path', 'fold', 'classID', 'class'
    """
    csv_path = os.path.join(dataset_root, "metadata", "UrbanSound8K.csv")
    df = pd.read_csv(csv_path)
    df["file_path"] = df.apply(
        lambda row: os.path.join(
            dataset_root, "audio", f"fold{row['fold']}", row["slice_file_name"]
        ),
        axis=1,
    )
    return df


# ------------------------------------------------------------------
# Visualisation
# ------------------------------------------------------------------

def plot_waveform(waveform: np.ndarray,
                  sr: int = SAMPLE_RATE,
                  title: str = "Waveform",
                  ax: plt.Axes | None = None) -> plt.Axes:
    """Plot the time-domain waveform."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 2))
    time = np.linspace(0, len(waveform) / sr, num=len(waveform))
    ax.plot(time, waveform, linewidth=0.6, color="#2196F3")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.set_xlim([0, time[-1]])
    return ax


def plot_spectrogram(waveform: np.ndarray,
                     sr: int = SAMPLE_RATE,
                     title: str = "Mel Spectrogram",
                     n_mels: int = 128,
                     hop_length: int = 512,
                     ax: plt.Axes | None = None) -> plt.Axes:
    """Plot the Mel spectrogram (dB scale)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 3))
    mel = librosa.feature.melspectrogram(
        y=waveform, sr=sr, n_mels=n_mels, hop_length=hop_length
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    img = librosa.display.specshow(
        mel_db, sr=sr, hop_length=hop_length,
        x_axis="time", y_axis="mel", ax=ax
    )
    plt.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title(title)
    return ax


def plot_sample(file_path: str,
                class_name: str = "",
                sr: int = SAMPLE_RATE) -> None:
    """
    Display waveform + mel spectrogram for a single audio file.
    """
    waveform, sr = load_audio(file_path, sr=sr)
    fig = plt.figure(figsize=(12, 5))
    gs  = gridspec.GridSpec(2, 1, hspace=0.4)

    ax1 = fig.add_subplot(gs[0])
    plot_waveform(waveform, sr=sr,
                  title=f"Waveform  —  {class_name}", ax=ax1)

    ax2 = fig.add_subplot(gs[1])
    plot_spectrogram(waveform, sr=sr,
                     title=f"Mel Spectrogram  —  {class_name}", ax=ax2)

    plt.suptitle(os.path.basename(file_path), fontsize=10, color="gray")
    plt.show()


def explore_dataset(df: pd.DataFrame,
                    dataset_root: str,
                    n_per_class: int = 1) -> None:
    """
    Visualise waveform + spectrogram for `n_per_class` examples per class,
    and play the audio in a Jupyter environment.

    Parameters
    ----------
    df           : metadata DataFrame (from load_metadata)
    dataset_root : root directory of UrbanSound8K
    n_per_class  : number of examples to show per class
    """
    for class_id, class_name in enumerate(CLASS_LABELS):
        subset = df[df["classID"] == class_id].head(n_per_class)
        for _, row in subset.iterrows():
            fp = row["file_path"]
            print(f"\n{'='*60}")
            print(f"Class {class_id}: {class_name}  |  {row['slice_file_name']}")
            print(f"Fold: {row['fold']}  |  Duration: {row['end'] - row['start']:.2f}s")
            plot_sample(fp, class_name=class_name)
            # Audio playback (works in Jupyter)
            try:
                waveform, sr = load_audio(fp)
                display(ipd.Audio(waveform, rate=sr))  # noqa: F821
            except Exception:
                pass


# ------------------------------------------------------------------
# Dataset summary
# ------------------------------------------------------------------

def summarise_dataset(df: pd.DataFrame) -> None:
    """Print a quick summary of the dataset distribution."""
    print("UrbanSound8K — Dataset Summary")
    print("=" * 45)
    print(f"Total clips : {len(df)}")
    print(f"\nSamples per class:")
    counts = df.groupby(["classID", "class"]).size().reset_index(name="count")
    for _, row in counts.iterrows():
        bar = "█" * (row["count"] // 30)
        print(f"  [{row['classID']:2d}] {row['class']:<20s} {row['count']:4d}  {bar}")

    print(f"\nSamples per fold:")
    for fold, cnt in df.groupby("fold").size().items():
        split = ("TRAIN" if fold in TRAIN_FOLDS
                 else "VAL"   if fold in VAL_FOLDS
                 else "TEST")
        print(f"  Fold {fold:2d} ({split:<5s}): {cnt}")
