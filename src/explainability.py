"""
src/explainability.py
─────────────────────
Model interpretability module.

  1. Attention weight visualisation
     – Heatmap of which timesteps the model focuses on
     – Per-activity average attention maps
     – Signal overlay with attention highlights

  2. SHAP (SHapley Additive exPlanations)
     – Global feature importance (channel-level)
     – Summary plot

Usage
─────
    python src/explainability.py
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import seaborn as sns
import tensorflow as tf
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data_pipeline import HARDataPipeline, ACTIVITY_LABELS
from src.models import AttentionLayer

LABEL_NAMES  = list(ACTIVITY_LABELS.values())
CHANNEL_NAMES = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x","body_gyro_y","body_gyro_z",
    "total_acc_x","total_acc_y","total_acc_z",
]

MODELS_DIR   = Path("models_saved")
PLOTS_DIR    = Path("plots")
OUTPUTS_DIR  = Path("outputs")
PLOTS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  Attention extractor
# ══════════════════════════════════════════════════════════════════════════════

def build_attention_extractor(model: tf.keras.Model) -> tf.keras.Model:
    """
    Build a sub-model that returns:
      (final_output, attention_weights)
    """
    attn_layer = model.get_layer("attention")
    lstm_out   = model.get_layer("lstm").output   # (batch, T', features)

    # Compute score and weights manually (mirrors AttentionLayer.call)
    score   = tf.nn.tanh(tf.matmul(lstm_out, attn_layer.W) + attn_layer.b)
    weights = tf.nn.softmax(score, axis=1)         # (batch, T', 1)
    context = tf.reduce_sum(weights * lstm_out, axis=1)

    # Re-attach classifier head layers after attention
    x = context
    for layer in model.layers:
        if layer.name in ("attention",):
            continue
        # find the dense layers that come after attention in the original model
        if isinstance(layer, (tf.keras.layers.Dense,
                               tf.keras.layers.Dropout,
                               tf.keras.layers.BatchNormalization)):
            try:
                x = layer(x)
            except Exception:
                pass

    extractor = tf.keras.Model(
        inputs=model.input,
        outputs=[model.output, weights],
        name="attn_extractor",
    )
    return extractor


# ══════════════════════════════════════════════════════════════════════════════
#  Attention heatmaps
# ══════════════════════════════════════════════════════════════════════════════

def get_attention_weights(model: tf.keras.Model, X: np.ndarray) -> np.ndarray:
    """
    Forward pass through model and retrieve cached attention weights.

    Returns
    -------
    weights : (n_samples, T', 1)  where T' = timesteps after pooling
    """
    attn_layer = model.get_layer("attention")
    lstm_layer = model.get_layer("lstm")

    # Intermediate model up to attention layer
    inter = tf.keras.Model(
        inputs  = model.input,
        outputs = lstm_layer.output,
    )
    lstm_out = inter.predict(X, verbose=0)          # (n, T', feat)

    W = attn_layer.W.numpy()
    b = attn_layer.b.numpy()
    score   = np.tanh(np.matmul(lstm_out, W) + b)  # (n, T', 1)
    weights = np.exp(score) / np.sum(np.exp(score), axis=1, keepdims=True)
    return weights.squeeze(-1)                       # (n, T')


def plot_attention_heatmap(
    model: tf.keras.Model,
    X: np.ndarray,
    y: np.ndarray,
    n_examples: int = 5,
):
    """
    For each activity class, plot the input signal overlaid with
    the attention weight heatmap.
    """
    weights = get_attention_weights(model, X)   # (n, T')
    T_orig  = X.shape[1]                        # 128
    T_attn  = weights.shape[1]                  # after pooling (~32)

    fig, axes = plt.subplots(len(LABEL_NAMES), 2, figsize=(16, 3 * len(LABEL_NAMES)))
    fig.suptitle("Attention Weight Visualisation per Activity",
                 fontsize=14, fontweight="bold")

    for cls_idx, cls_name in enumerate(LABEL_NAMES):
        indices = np.where(y == cls_idx)[0]
        if len(indices) == 0:
            continue
        idx = indices[0]

        # Raw signal (first channel)
        signal = X[idx, :, 0]
        attn   = weights[idx]                   # (T',)

        # Up-sample attention weights to match signal length
        attn_up = np.interp(
            np.linspace(0, T_attn - 1, T_orig),
            np.arange(T_attn),
            attn,
        )

        ax_sig  = axes[cls_idx, 0]
        ax_attn = axes[cls_idx, 1]

        ax_sig.plot(signal, lw=1.5, color="steelblue")
        ax_sig.fill_between(
            range(T_orig), signal.min(),
            signal.min() + (signal.max() - signal.min()) * attn_up,
            alpha=0.35, color="orange",
        )
        ax_sig.set_title(f"{cls_name} — Signal + Attention", fontsize=9)
        ax_sig.set_xlim(0, T_orig)
        ax_sig.grid(alpha=0.2)

        im = ax_attn.imshow(
            attn[np.newaxis, :],
            aspect="auto", cmap="hot",
            extent=[0, T_attn, 0, 1],
        )
        ax_attn.set_title(f"{cls_name} — Attention Weights", fontsize=9)
        ax_attn.set_xlabel("Timestep (compressed)")
        ax_attn.set_yticks([])
        plt.colorbar(im, ax=ax_attn, orientation="horizontal", pad=0.35, shrink=0.8)

    plt.tight_layout()
    fout = PLOTS_DIR / "attention_heatmaps.png"
    plt.savefig(fout, dpi=150)
    plt.close()
    print(f"  [Attention] Saved → {fout}")


def plot_average_attention_per_class(
    model: tf.keras.Model,
    X: np.ndarray,
    y: np.ndarray,
):
    """Average attention weights per activity class (bar chart)."""
    weights = get_attention_weights(model, X)
    T_attn  = weights.shape[1]

    avg_weights = np.zeros((len(LABEL_NAMES), T_attn))
    for cls_idx in range(len(LABEL_NAMES)):
        mask = (y == cls_idx)
        if mask.sum() > 0:
            avg_weights[cls_idx] = weights[mask].mean(axis=0)

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(avg_weights, aspect="auto", cmap="YlOrRd")
    ax.set_yticks(range(len(LABEL_NAMES)))
    ax.set_yticklabels(LABEL_NAMES)
    ax.set_xlabel("Compressed Timestep (post-pooling)")
    ax.set_title("Average Attention Weights per Activity Class",
                 fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fout = PLOTS_DIR / "attention_per_class.png"
    plt.savefig(fout, dpi=150)
    plt.close()
    print(f"  [Avg attention] Saved → {fout}")


# ══════════════════════════════════════════════════════════════════════════════
#  SHAP feature importance (channel-level)
# ══════════════════════════════════════════════════════════════════════════════

def shap_channel_importance(
    model: tf.keras.Model,
    X_background: np.ndarray,
    X_explain: np.ndarray,
    n_background: int = 100,
    n_explain: int    = 200,
):
    """
    Compute SHAP values at the channel (feature) level by averaging over time.

    Uses GradientExplainer (faster than KernelExplainer for neural nets).
    """
    try:
        import shap
    except ImportError:
        print("  [SHAP] skipped — run 'pip install shap' to enable.")
        return

    print("  [SHAP] Computing GradientExplainer values…")

    bg = X_background[:n_background]
    ex = X_explain[:n_explain]

    explainer  = shap.GradientExplainer(model, bg)
    shap_vals  = explainer.shap_values(ex)         # list of (n, T, C) per class

    # Mean |SHAP| across time and samples for each channel
    mean_abs = np.mean([np.abs(sv).mean(axis=(0, 1)) for sv in shap_vals], axis=0)
    # mean_abs : (n_channels,)

    df = pd.DataFrame({
        "Channel": CHANNEL_NAMES,
        "Mean |SHAP|": mean_abs,
    }).sort_values("Mean |SHAP|", ascending=False)

    fig, ax = plt.subplots(figsize=(9, 4))
    sns.barplot(data=df, x="Mean |SHAP|", y="Channel",
                palette="viridis", ax=ax)
    ax.set_title("SHAP Channel Importance (averaged over time & classes)",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Mean |SHAP Value|")
    plt.tight_layout()
    fout = PLOTS_DIR / "shap_channel_importance.png"
    plt.savefig(fout, dpi=150)
    plt.close()
    print(f"  [SHAP] Saved → {fout}")

    df.to_csv(OUTPUTS_DIR / "shap_channel_importance.csv", index=False)
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    data_root = "data/UCI HAR Dataset"
    dp        = HARDataPipeline(data_root=data_root)
    _, _, X_test, y_test = dp.load()

    model_name = "CNN_LSTM_Attention"
    model_path = MODELS_DIR / f"{model_name}.keras"
    if not model_path.exists():
        print(f"[ERROR] {model_name}.keras not found. Run train.py first.")
        return

    model = tf.keras.models.load_model(
        model_path, custom_objects={"AttentionLayer": AttentionLayer}
    )
    print(f"[OK] Loaded {model_name}")

    print("\n[1/3] Attention heatmaps per activity…")
    plot_attention_heatmap(model, X_test[:300], y_test[:300])

    print("\n[2/3] Average attention per class…")
    plot_average_attention_per_class(model, X_test, y_test)

    print("\n[3/3] SHAP channel importance…")
    shap_channel_importance(model, X_test, X_test)


if __name__ == "__main__":
    main()
