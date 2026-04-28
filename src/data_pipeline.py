"""
src/data_pipeline.py
────────────────────
Professional-grade data pipeline for UCI HAR dataset.

Pipeline stages
───────────────
1. Load raw sensor signals (9 channels × 128 timesteps)
2. StandardScaler normalisation
3. Optional sliding-window re-segmentation (for raw-signal datasets)
4. Noise injection for robustness evaluation
5. Label encoding + one-hot conversion

Usage
─────
    from src.data_pipeline import HARDataPipeline
    dp = HARDataPipeline(data_root="data/UCI HAR Dataset")
    X_train, y_train, X_test, y_test = dp.load()
    X_noisy = dp.inject_noise(X_test, sigma=0.05)
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

# ─── Label map ────────────────────────────────────────────────────────────────
ACTIVITY_LABELS = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING",
}

N_TIMESTEPS = 128
N_FEATURES  = 9          # 9 inertial signal channels
N_CLASSES   = 6


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _load_signals(split_dir: Path, subset: str) -> np.ndarray:
    """Load all 9 raw inertial signals for a given split (train/test)."""
    signal_names = [
        "body_acc_x", "body_acc_y", "body_acc_z",
        "body_gyro_x", "body_gyro_y", "body_gyro_z",
        "total_acc_x", "total_acc_y", "total_acc_z",
    ]
    signals = []
    for name in signal_names:
        fpath = split_dir / "Inertial Signals" / f"{name}_{subset}.txt"
        data  = np.loadtxt(fpath)          # (n_samples, 128)
        signals.append(data)
    # Stack → (n_samples, 128, 9)
    return np.stack(signals, axis=2)


def _load_labels(split_dir: Path, subset: str) -> np.ndarray:
    fpath = split_dir / f"y_{subset}.txt"
    return np.loadtxt(fpath, dtype=int) - 1  # 0-indexed


# ─── Pipeline class ───────────────────────────────────────────────────────────

class HARDataPipeline:
    """
    End-to-end data pipeline for Human Activity Recognition.

    Parameters
    ----------
    data_root : str
        Path to extracted 'UCI HAR Dataset' folder.
    window_size : int
        Sliding window length (samples).
    overlap : float
        Overlap ratio for sliding window (0–1).
    scaler_path : str | None
        Where to save / load the fitted StandardScaler.
    """

    def __init__(
        self,
        data_root: str = "data/UCI HAR Dataset",
        window_size: int = 128,
        overlap: float = 0.5,
        scaler_path: str = "models_saved/scaler.pkl",
    ):
        self.data_root   = Path(data_root)
        self.window_size = window_size
        self.overlap     = overlap
        self.scaler_path = scaler_path
        self.scaler      = StandardScaler()
        self.label_names = list(ACTIVITY_LABELS.values())

    # ── Public API ─────────────────────────────────────────────────────────────

    def load(self, normalise: bool = True):
        """
        Load, (optionally) normalise and return train/test splits.

        Returns
        -------
        X_train, y_train, X_test, y_test  (all np.ndarray)
        X shape : (n_samples, 128, 9)
        y shape : (n_samples,)  — integer class labels 0-5
        """
        print("[Pipeline] Loading raw signals…")
        X_train = _load_signals(self.data_root / "train", "train")
        y_train = _load_labels(self.data_root / "train", "train")
        X_test  = _load_signals(self.data_root / "test",  "test")
        y_test  = _load_labels(self.data_root / "test",   "test")

        print(f"  Train: {X_train.shape}  |  Test: {X_test.shape}")

        if normalise:
            X_train, X_test = self._normalise(X_train, X_test)

        return X_train, y_train, X_test, y_test

    def sliding_window(self, X: np.ndarray, y: np.ndarray):
        """
        Apply sliding-window segmentation to long continuous signals.

        Useful when working with raw, un-windowed sensor streams.
        UCI HAR is already pre-windowed (128 steps), so this is provided
        for the real-time simulation engine and WISDM compatibility.

        Parameters
        ----------
        X : (n_samples, signal_length, n_channels)
        y : (n_samples,)

        Returns
        -------
        X_win : (n_windows, window_size, n_channels)
        y_win : (n_windows,)
        """
        step    = int(self.window_size * (1 - self.overlap))
        X_wins, y_wins = [], []

        for xi, yi in zip(X, y):
            n = xi.shape[0]
            for start in range(0, n - self.window_size + 1, step):
                X_wins.append(xi[start : start + self.window_size])
                y_wins.append(yi)

        return np.array(X_wins), np.array(y_wins)

    @staticmethod
    def inject_noise(X: np.ndarray, sigma: float = 0.05) -> np.ndarray:
        """
        Additive Gaussian noise injection for robustness testing.

        Parameters
        ----------
        X     : input array (any shape)
        sigma : standard deviation of noise

        Returns
        -------
        X_noisy : noisy copy of X
        """
        noise = np.random.normal(0, sigma, X.shape)
        return X + noise

    @staticmethod
    def to_onehot(y: np.ndarray, n_classes: int = N_CLASSES) -> np.ndarray:
        """Convert integer labels to one-hot vectors."""
        import tensorflow as tf
        return tf.keras.utils.to_categorical(y, num_classes=n_classes)

    def get_label_name(self, idx: int) -> str:
        return self.label_names[idx]

    def get_all_label_names(self):
        return self.label_names

    # ── Private helpers ────────────────────────────────────────────────────────

    def _normalise(self, X_train: np.ndarray, X_test: np.ndarray):
        """Fit StandardScaler on train, apply to both splits."""
        print("[Pipeline] Normalising signals (StandardScaler)…")
        n_train, T, C = X_train.shape
        n_test         = X_test.shape[0]

        # Reshape to 2-D for scaler, then back
        self.scaler.fit(X_train.reshape(-1, C))
        X_train_n = self.scaler.transform(X_train.reshape(-1, C)).reshape(n_train, T, C)
        X_test_n  = self.scaler.transform(X_test.reshape(-1, C)).reshape(n_test,  T, C)

        # Persist scaler
        os.makedirs(os.path.dirname(self.scaler_path), exist_ok=True)
        joblib.dump(self.scaler, self.scaler_path)
        print(f"  Scaler saved → {self.scaler_path}")

        return X_train_n, X_test_n

    def load_scaler(self):
        self.scaler = joblib.load(self.scaler_path)
        return self.scaler


# ─── Standalone test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    dp = HARDataPipeline()
    Xtr, ytr, Xte, yte = dp.load()
    print("Train:", Xtr.shape, ytr.shape)
    print("Test: ", Xte.shape, yte.shape)

    Xn = HARDataPipeline.inject_noise(Xte)
    print("Noisy:", Xn.shape, "| mean Δ:", np.abs(Xn - Xte).mean().round(4))
