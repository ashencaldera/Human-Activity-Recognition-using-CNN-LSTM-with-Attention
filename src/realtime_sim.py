"""
src/realtime_sim.py
────────────────────
Real-time prediction simulation engine.

Mimics a live sensor stream by:
  1. Sliding a window across a test sequence
  2. Feeding each window to the model
  3. Returning timestamped predictions + confidence

Can be imported by the Streamlit app or run standalone.

Usage
─────
    python src/realtime_sim.py
"""

import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import tensorflow as tf
from pathlib import Path
from collections import deque

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data_pipeline import HARDataPipeline, ACTIVITY_LABELS
from src.models import AttentionLayer

LABEL_NAMES  = list(ACTIVITY_LABELS.values())
MODELS_DIR   = Path("models_saved")
PLOTS_DIR    = Path("plots")
OUTPUTS_DIR  = Path("outputs")
PLOTS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  Core Streaming Engine
# ══════════════════════════════════════════════════════════════════════════════

class RealTimePredictor:
    """
    Simulates real-time Human Activity Recognition.

    Parameters
    ----------
    model       : trained tf.keras.Model
    window_size : int      — same window used during training (128)
    step_size   : int      — samples between consecutive predictions
                             (step_size = window_size → no overlap,
                              step_size = 1          → maximum smoothness)
    label_names : list[str]
    """

    def __init__(
        self,
        model: tf.keras.Model,
        window_size: int = 128,
        step_size: int   = 32,
        label_names: list = None,
    ):
        self.model       = model
        self.window_size = window_size
        self.step_size   = step_size
        self.label_names = label_names or LABEL_NAMES
        self.buffer      = deque(maxlen=window_size)   # ring buffer

    # ── Batch stream (offline simulation) ─────────────────────────────────────

    def predict_stream(
        self,
        X: np.ndarray,
        add_noise: bool = False,
        noise_sigma: float = 0.05,
        sleep_ms: float = 0.0,
    ) -> pd.DataFrame:
        """
        Slide a window over X and return a DataFrame of predictions.

        Parameters
        ----------
        X          : (n_samples, window_size, n_features)  — test data
        add_noise  : bool   — inject Gaussian noise to simulate sensor degradation
        noise_sigma: float  — noise std-dev
        sleep_ms   : float  — simulated latency per prediction (ms)

        Returns
        -------
        DataFrame with columns:
          sample_idx | true_label | pred_label | confidence | <per-class prob>
        """
        if add_noise:
            X = HARDataPipeline.inject_noise(X, sigma=noise_sigma)

        rows = []
        n    = len(X)

        for i in range(0, n, self.step_size):
            sample = X[i][np.newaxis, ...]           # (1, 128, 9)
            prob   = self.model.predict(sample, verbose=0)[0]   # (n_classes,)
            pred   = int(np.argmax(prob))
            conf   = float(prob[pred])

            row = {
                "sample_idx": i,
                "pred_label": self.label_names[pred],
                "pred_idx":   pred,
                "confidence": conf,
            }
            for j, lname in enumerate(self.label_names):
                row[f"prob_{lname}"] = float(prob[j])

            rows.append(row)

            if sleep_ms > 0:
                time.sleep(sleep_ms / 1000.0)

        return pd.DataFrame(rows)

    # ── Online single-sample update (for live integration) ────────────────────

    def update(self, new_sample: np.ndarray) -> dict | None:
        """
        Push one new sensor frame (shape: (n_features,)) into the ring buffer.
        Returns a prediction dict once the buffer is full; otherwise None.
        """
        self.buffer.append(new_sample)
        if len(self.buffer) < self.window_size:
            return None

        window = np.array(self.buffer)[np.newaxis, ...]    # (1, 128, n_features)
        prob   = self.model.predict(window, verbose=0)[0]
        pred   = int(np.argmax(prob))
        return {
            "pred_label":  self.label_names[pred],
            "pred_idx":    pred,
            "confidence":  float(prob[pred]),
            "probabilities": {n: float(p) for n, p in zip(self.label_names, prob)},
        }


# ══════════════════════════════════════════════════════════════════════════════
#  Visualise streaming results
# ══════════════════════════════════════════════════════════════════════════════

def plot_stream_predictions(df: pd.DataFrame, model_name: str, n_samples: int = 200):
    """
    Plots:
      (a) Predicted activity over time (coloured dots)
      (b) Confidence over time
    """
    sub = df.head(n_samples)

    activity_map   = {l: i for i, l in enumerate(LABEL_NAMES)}
    numeric_pred   = sub["pred_label"].map(activity_map)
    colours        = plt.cm.tab10(np.linspace(0, 1, len(LABEL_NAMES)))

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    fig.suptitle(f"{model_name} — Real-Time Prediction Simulation",
                 fontsize=13, fontweight="bold")

    # Activity timeline
    for idx, row in sub.iterrows():
        c = colours[activity_map[row["pred_label"]]]
        axes[0].scatter(row["sample_idx"], activity_map[row["pred_label"]],
                        color=c, s=20)
    axes[0].set_yticks(range(len(LABEL_NAMES)))
    axes[0].set_yticklabels(LABEL_NAMES, fontsize=8)
    axes[0].set_ylabel("Predicted Activity")
    axes[0].grid(alpha=0.25)

    # Confidence
    axes[1].fill_between(sub["sample_idx"], sub["confidence"],
                         alpha=0.4, color="steelblue")
    axes[1].plot(sub["sample_idx"], sub["confidence"],
                 lw=1.5, color="steelblue")
    axes[1].axhline(0.9, ls="--", color="green",  alpha=0.6, label="90% conf.")
    axes[1].axhline(0.7, ls="--", color="orange", alpha=0.6, label="70% conf.")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("Confidence")
    axes[1].set_xlabel("Sample Index")
    axes[1].legend(loc="lower right")
    axes[1].grid(alpha=0.25)

    plt.tight_layout()
    fout = PLOTS_DIR / f"{model_name}_stream.png"
    plt.savefig(fout, dpi=150)
    plt.close()
    print(f"  [Stream plot] Saved → {fout}")


def plot_activity_distribution(df: pd.DataFrame, model_name: str):
    """Pie chart of predicted activity frequencies."""
    counts = df["pred_label"].value_counts()
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%",
           startangle=90, colors=plt.cm.tab10.colors)
    ax.set_title(f"{model_name} — Predicted Activity Distribution",
                 fontweight="bold")
    plt.tight_layout()
    fout = PLOTS_DIR / f"{model_name}_activity_dist.png"
    plt.savefig(fout, dpi=150)
    plt.close()
    print(f"  [Activity dist] Saved → {fout}")


# ══════════════════════════════════════════════════════════════════════════════
#  Latency benchmark
# ══════════════════════════════════════════════════════════════════════════════

def benchmark_latency(model, X_test, n_trials: int = 100) -> dict:
    """
    Measure single-sample inference latency (ms).
    """
    sample = X_test[:1]
    _ = model.predict(sample, verbose=0)   # warm-up

    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        model.predict(sample, verbose=0)
        times.append((time.perf_counter() - t0) * 1000)

    return {
        "mean_ms":   round(float(np.mean(times)),   2),
        "std_ms":    round(float(np.std(times)),    2),
        "min_ms":    round(float(np.min(times)),    2),
        "max_ms":    round(float(np.max(times)),    2),
        "p95_ms":    round(float(np.percentile(times, 95)), 2),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # Load data
    data_root = "data/UCI HAR Dataset"
    dp        = HARDataPipeline(data_root=data_root)
    _, _, X_test, y_test = dp.load()

    # Use best model: CNN_LSTM_Attention
    model_name = "CNN_LSTM_Attention"
    model_path = MODELS_DIR / f"{model_name}.keras"
    if not model_path.exists():
        print(f"[ERROR] Trained model not found at {model_path}.")
        print("        Run 'python src/train.py' first.")
        return

    model = tf.keras.models.load_model(
        model_path, custom_objects={"AttentionLayer": AttentionLayer}
    )
    print(f"[OK] Loaded {model_name}")

    # Simulate stream
    predictor = RealTimePredictor(model, window_size=128, step_size=32)

    print("\n[INFO] Running clean stream simulation…")
    df_clean = predictor.predict_stream(X_test[:500])
    df_clean.to_csv(OUTPUTS_DIR / f"{model_name}_stream_clean.csv", index=False)
    plot_stream_predictions(df_clean, f"{model_name}_clean")
    plot_activity_distribution(df_clean, f"{model_name}_clean")

    print("\n[INFO] Running noisy stream simulation (σ=0.05)…")
    df_noisy = predictor.predict_stream(X_test[:500], add_noise=True, noise_sigma=0.05)
    df_noisy.to_csv(OUTPUTS_DIR / f"{model_name}_stream_noisy.csv", index=False)
    plot_stream_predictions(df_noisy, f"{model_name}_noisy")

    # Latency
    print("\n[INFO] Benchmarking inference latency…")
    lat = benchmark_latency(model, X_test)
    print(f"  Mean: {lat['mean_ms']} ms  |  p95: {lat['p95_ms']} ms  |  "
          f"Std: {lat['std_ms']} ms")

    # Save latency
    pd.DataFrame([lat]).to_csv(OUTPUTS_DIR / "latency_benchmark.csv", index=False)


if __name__ == "__main__":
    main()
