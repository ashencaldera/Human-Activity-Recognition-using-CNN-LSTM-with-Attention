"""
src/train.py
────────────
Trains all four models with professional callbacks and saves:
  • model weights        → models_saved/<name>.keras
  • training history     → outputs/<name>_history.npy
  • training plots       → plots/<name>_training.png

Usage
─────
    python src/train.py [--model all | ANN | LSTM | CNN_LSTM | CNN_LSTM_Attention]

Environment
───────────
    Set DATA_ROOT env-var if UCI HAR is not at 'data/UCI HAR Dataset'.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import tensorflow as tf
from pathlib import Path

# ── allow running from project root ──────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data_pipeline import HARDataPipeline
from src.models import build_model, MODEL_REGISTRY

# ── dirs ──────────────────────────────────────────────────────────────────────
MODELS_DIR  = Path("models_saved")
OUTPUTS_DIR = Path("outputs")
PLOTS_DIR   = Path("plots")
for d in (MODELS_DIR, OUTPUTS_DIR, PLOTS_DIR):
    d.mkdir(exist_ok=True)

# ── training hyper-params ─────────────────────────────────────────────────────
EPOCHS     = 50
BATCH_SIZE = 64
LR         = 1e-3


# ══════════════════════════════════════════════════════════════════════════════
#  Callbacks
# ══════════════════════════════════════════════════════════════════════════════

def get_callbacks(model_name: str):
    ckpt_path = str(MODELS_DIR / f"{model_name}_best.keras")
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=ckpt_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=0,
        ),
        tf.keras.callbacks.CSVLogger(
            str(OUTPUTS_DIR / f"{model_name}_log.csv"),
            append=False,
        ),
    ]


# ══════════════════════════════════════════════════════════════════════════════
#  Plot training curves
# ══════════════════════════════════════════════════════════════════════════════

def plot_history(history, model_name: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{model_name} — Training History", fontsize=14, fontweight="bold")

    # Loss
    axes[0].plot(history["loss"],     label="Train Loss",  lw=2)
    axes[0].plot(history["val_loss"], label="Val Loss",    lw=2, ls="--")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Sparse Categorical Cross-Entropy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Accuracy
    axes[1].plot(history["accuracy"],     label="Train Acc", lw=2)
    axes[1].plot(history["val_accuracy"], label="Val Acc",   lw=2, ls="--")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out = PLOTS_DIR / f"{model_name}_training.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  [Plot] Saved → {out}")


# ══════════════════════════════════════════════════════════════════════════════
#  Train one model
# ══════════════════════════════════════════════════════════════════════════════

def train_model(model_name: str, X_train, y_train, X_val, y_val):
    print(f"\n{'='*60}")
    print(f"  Training: {model_name}")
    print(f"{'='*60}")

    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_model(model_name, input_shape=input_shape)
    model.summary(print_fn=lambda x: None)   # silent

    hist = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=get_callbacks(model_name),
        verbose=2,
    )

    # Save history
    h_path = OUTPUTS_DIR / f"{model_name}_history.npy"
    np.save(h_path, hist.history)
    print(f"  [History] Saved → {h_path}")

    # Save final model
    m_path = MODELS_DIR / f"{model_name}.keras"
    model.save(m_path)
    print(f"  [Model] Saved → {m_path}")

    # Plot
    plot_history(hist.history, model_name)

    return model, hist.history


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main(models_to_train="all"):
    # ── Load data ──────────────────────────────────────────────────────────
    data_root = os.environ.get("DATA_ROOT", "data/UCI HAR Dataset")
    dp = HARDataPipeline(data_root=data_root)
    X_train_full, y_train_full, X_test, y_test = dp.load()

    # Hold out 20% of train as validation
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=0.2,
        random_state=42,
        stratify=y_train_full,
    )
    print(f"Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")

    # ── Select models ──────────────────────────────────────────────────────
    if models_to_train == "all":
        names = list(MODEL_REGISTRY.keys())
    else:
        names = [models_to_train]

    results = {}
    for name in names:
        model, history = train_model(name, X_train, y_train, X_val, y_val)

        # Quick test-set evaluation
        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        results[name] = {"test_accuracy": acc, "test_loss": loss}
        print(f"  [{name}] Test Acc: {acc:.4f}  |  Test Loss: {loss:.4f}")

    # ── Summary table ──────────────────────────────────────────────────────
    print("\n" + "=" * 45)
    print(f"  {'Model':<25}  {'Test Acc':>10}")
    print("-" * 45)
    for name, r in results.items():
        print(f"  {name:<25}  {r['test_accuracy']:>10.4f}")
    print("=" * 45)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default="all",
        help="Model to train: all | ANN | LSTM | CNN_LSTM | CNN_LSTM_Attention"
    )
    args = parser.parse_args()
    main(args.model)
