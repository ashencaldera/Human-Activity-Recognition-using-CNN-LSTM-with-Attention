"""
src/evaluate.py
───────────────
Professional evaluation suite.

  1. Per-model metrics  (Accuracy, Precision, Recall, F1)
  2. Confusion matrix   (absolute + normalised)
  3. Robustness test    (clean vs noisy at multiple σ levels)
  4. Model comparison table
  5. Error analysis     (most-confused class pairs)
  6. All plots saved to plots/

Usage
─────
    python src/evaluate.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data_pipeline import HARDataPipeline, ACTIVITY_LABELS
from src.models import MODEL_REGISTRY, AttentionLayer

MODELS_DIR  = Path("models_saved")
PLOTS_DIR   = Path("plots")
OUTPUTS_DIR = Path("outputs")
PLOTS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

LABEL_NAMES = list(ACTIVITY_LABELS.values())
NOISE_LEVELS = [0.0, 0.02, 0.05, 0.10, 0.20]


# ══════════════════════════════════════════════════════════════════════════════
#  Load trained model
# ══════════════════════════════════════════════════════════════════════════════

def load_model(name: str) -> tf.keras.Model:
    path = MODELS_DIR / f"{name}.keras"
    if not path.exists():
        path = MODELS_DIR / f"{name}_best.keras"
    return tf.keras.models.load_model(
        path, custom_objects={"AttentionLayer": AttentionLayer}
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Core metrics
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_model(model, X, y, label: str = "") -> dict:
    y_pred_prob = model.predict(X, verbose=0)
    y_pred      = np.argmax(y_pred_prob, axis=1)

    return {
        "label":     label,
        "accuracy":  accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, average="weighted", zero_division=0),
        "recall":    recall_score(y, y_pred, average="weighted", zero_division=0),
        "f1":        f1_score(y, y_pred, average="weighted", zero_division=0),
        "y_pred":    y_pred,
        "y_true":    y,
        "y_prob":    y_pred_prob,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Confusion matrix
# ══════════════════════════════════════════════════════════════════════════════

def plot_confusion_matrix(y_true, y_pred, model_name: str, normalise: bool = True):
    cm = confusion_matrix(y_true, y_pred)
    if normalise:
        cm_plot = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt, vmax = ".2f", 1.0
        title = f"{model_name} — Normalised Confusion Matrix"
    else:
        cm_plot, fmt, vmax = cm, "d", None
        title = f"{model_name} — Confusion Matrix"

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm_plot,
        annot=True, fmt=fmt,
        xticklabels=LABEL_NAMES,
        yticklabels=LABEL_NAMES,
        cmap="Blues",
        vmin=0, vmax=vmax,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    tag  = "norm" if normalise else "abs"
    fout = PLOTS_DIR / f"{model_name}_cm_{tag}.png"
    plt.savefig(fout, dpi=150)
    plt.close()
    print(f"  [CM] Saved → {fout}")


# ══════════════════════════════════════════════════════════════════════════════
#  Per-class F1 bar chart
# ══════════════════════════════════════════════════════════════════════════════

def plot_per_class_f1(y_true, y_pred, model_name: str):
    f1s = f1_score(y_true, y_pred, average=None, zero_division=0)
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(LABEL_NAMES, f1s, color=sns.color_palette("Blues_d", len(LABEL_NAMES)))
    ax.set_ylim(0, 1.05)
    ax.set_title(f"{model_name} — Per-Class F1 Score", fontweight="bold")
    ax.set_ylabel("F1 Score")
    ax.set_xlabel("Activity")
    for bar, val in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    fout = PLOTS_DIR / f"{model_name}_per_class_f1.png"
    plt.savefig(fout, dpi=150)
    plt.close()
    print(f"  [F1 chart] Saved → {fout}")


# ══════════════════════════════════════════════════════════════════════════════
#  Robustness analysis
# ══════════════════════════════════════════════════════════════════════════════

def robustness_test(model, X_test, y_test, model_name: str) -> pd.DataFrame:
    rows = []
    for sigma in NOISE_LEVELS:
        X_noisy = HARDataPipeline.inject_noise(X_test, sigma=sigma) if sigma > 0 else X_test
        res     = evaluate_model(model, X_noisy, y_test, label=f"σ={sigma}")
        rows.append({
            "Noise σ":  sigma,
            "Accuracy": round(res["accuracy"],  4),
            "F1":       round(res["f1"],        4),
            "Precision":round(res["precision"], 4),
            "Recall":   round(res["recall"],    4),
        })
        print(f"    σ={sigma:.2f}  Acc={res['accuracy']:.4f}  F1={res['f1']:.4f}")

    df = pd.DataFrame(rows)

    # ── Plot ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["Noise σ"], df["Accuracy"], "o-", lw=2, label="Accuracy")
    ax.plot(df["Noise σ"], df["F1"],       "s--",lw=2, label="F1")
    ax.set_title(f"{model_name} — Robustness Under Gaussian Noise", fontweight="bold")
    ax.set_xlabel("Noise σ")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fout = PLOTS_DIR / f"{model_name}_robustness.png"
    plt.savefig(fout, dpi=150)
    plt.close()
    print(f"  [Robustness] Saved → {fout}")

    return df


# ══════════════════════════════════════════════════════════════════════════════
#  Model comparison
# ══════════════════════════════════════════════════════════════════════════════

def comparison_table(results: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame([{
        "Model":     r["label"],
        "Accuracy":  round(r["accuracy"],  4),
        "Precision": round(r["precision"], 4),
        "Recall":    round(r["recall"],    4),
        "F1":        round(r["f1"],        4),
    } for r in results])

    print("\n" + "=" * 65)
    print(df.to_string(index=False))
    print("=" * 65)

    # ── Grouped bar chart ──────────────────────────────────────────────────
    metrics = ["Accuracy", "Precision", "Recall", "F1"]
    x = np.arange(len(df))
    w = 0.2

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, metric in enumerate(metrics):
        ax.bar(x + i * w, df[metric], width=w, label=metric)

    ax.set_xticks(x + w * 1.5)
    ax.set_xticklabels(df["Model"], rotation=15, ha="right")
    ax.set_ylim(0.5, 1.02)
    ax.set_title("Model Comparison — Test Set Metrics", fontsize=13, fontweight="bold")
    ax.set_ylabel("Score")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fout = PLOTS_DIR / "model_comparison.png"
    plt.savefig(fout, dpi=150)
    plt.close()
    print(f"\n  [Comparison chart] Saved → {fout}")

    csv_out = OUTPUTS_DIR / "model_comparison.csv"
    df.to_csv(csv_out, index=False)
    print(f"  [Comparison CSV]   Saved → {csv_out}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  Error analysis
# ══════════════════════════════════════════════════════════════════════════════

def error_analysis(y_true, y_pred, model_name: str):
    """Print top confused class pairs and save a report."""
    cm = confusion_matrix(y_true, y_pred)
    np.fill_diagonal(cm, 0)   # zero out correct predictions

    pairs = []
    for i in range(len(LABEL_NAMES)):
        for j in range(len(LABEL_NAMES)):
            if cm[i, j] > 0:
                pairs.append({
                    "True":      LABEL_NAMES[i],
                    "Predicted": LABEL_NAMES[j],
                    "Count":     cm[i, j],
                })

    df = pd.DataFrame(pairs).sort_values("Count", ascending=False).head(10)
    print(f"\n  Top confused pairs [{model_name}]:")
    print(df.to_string(index=False))

    fout = OUTPUTS_DIR / f"{model_name}_error_analysis.csv"
    df.to_csv(fout, index=False)
    print(f"  [Error analysis] Saved → {fout}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    data_root = os.environ.get("DATA_ROOT", "data/UCI HAR Dataset")
    dp = HARDataPipeline(data_root=data_root)
    _, _, X_test, y_test = dp.load()

    all_results = []

    for name in MODEL_REGISTRY:
        model_path = MODELS_DIR / f"{name}.keras"
        if not model_path.exists():
            print(f"[SKIP] {name} — model file not found at {model_path}")
            continue

        print(f"\n{'─'*55}")
        print(f"  Evaluating: {name}")
        print(f"{'─'*55}")

        model = load_model(name)
        res   = evaluate_model(model, X_test, y_test, label=name)
        all_results.append(res)

        # Confusion matrix (both forms)
        plot_confusion_matrix(res["y_true"], res["y_pred"], name, normalise=True)
        plot_confusion_matrix(res["y_true"], res["y_pred"], name, normalise=False)

        # Per-class F1
        plot_per_class_f1(res["y_true"], res["y_pred"], name)

        # Classification report
        print("\n" + classification_report(
            res["y_true"], res["y_pred"],
            target_names=LABEL_NAMES, zero_division=0
        ))

        # Robustness
        print(f"  [Robustness test for {name}]")
        rob_df = robustness_test(model, X_test, y_test, name)
        rob_df.to_csv(OUTPUTS_DIR / f"{name}_robustness.csv", index=False)

        # Error analysis
        error_analysis(res["y_true"], res["y_pred"], name)

    # Final comparison table
    if all_results:
        comparison_table(all_results)


if __name__ == "__main__":
    main()
