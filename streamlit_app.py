"""
app/streamlit_app.py
─────────────────────
Streamlit deployment interface for Real-Time HAR.

Features
────────
  • Upload sensor CSV  → predict activities
  • Simulate live stream  → rolling window animation
  • Show: predicted activity, confidence score, signal graph
  • Robustness toggle  → add noise at user-defined level
  • Attention weight viewer  → see which timesteps matter
  • Model comparison metrics

Run
───
    streamlit run app/streamlit_app.py
"""

import sys
import io
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import streamlit as st
import tensorflow as tf
from pathlib import Path

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from src.data_pipeline import HARDataPipeline, ACTIVITY_LABELS
from src.models import AttentionLayer
from src.realtime_sim import RealTimePredictor, benchmark_latency
from src.explainability import get_attention_weights

LABEL_NAMES   = list(ACTIVITY_LABELS.values())
CHANNEL_NAMES = [
    "body_acc_x","body_acc_y","body_acc_z",
    "body_gyro_x","body_gyro_y","body_gyro_z",
    "total_acc_x","total_acc_y","total_acc_z",
]
MODELS_DIR    = ROOT / "models_saved"
DATA_ROOT     = ROOT / "data" / "UCI HAR Dataset"

ACTIVITY_EMOJIS = {
    "WALKING":           "🚶",
    "WALKING_UPSTAIRS":  "🏃‍♂️⬆️",
    "WALKING_DOWNSTAIRS":"🏃‍♂️⬇️",
    "SITTING":           "🪑",
    "STANDING":          "🧍",
    "LAYING":            "🛏️",
}

ACTIVITY_COLORS = {
    "WALKING":           "#2196F3",
    "WALKING_UPSTAIRS":  "#4CAF50",
    "WALKING_DOWNSTAIRS":"#FF9800",
    "SITTING":           "#9C27B0",
    "STANDING":          "#F44336",
    "LAYING":            "#795548",
}

# ══════════════════════════════════════════════════════════════════════════════
#  Cached resources
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_model(name: str):
    path = MODELS_DIR / f"{name}.keras"
    if not path.exists():
        return None
    return tf.keras.models.load_model(
        path, custom_objects={"AttentionLayer": AttentionLayer}
    )


@st.cache_data
def load_test_data():
    if not DATA_ROOT.exists():
        return None, None
    dp = HARDataPipeline(data_root=str(DATA_ROOT))
    _, _, X_test, y_test = dp.load()
    return X_test, y_test


@st.cache_data
def load_history(name: str):
    p = ROOT / "outputs" / f"{name}_history.npy"
    if not p.exists():
        return None
    return np.load(p, allow_pickle=True).item()


@st.cache_data
def load_comparison_csv():
    p = ROOT / "outputs" / "model_comparison.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


# ══════════════════════════════════════════════════════════════════════════════
#  Plot helpers (all return fig objects for st.pyplot)
# ══════════════════════════════════════════════════════════════════════════════

def fig_signal(X_sample: np.ndarray, title: str = "Sensor Signal") -> plt.Figure:
    fig, axes = plt.subplots(3, 3, figsize=(12, 6), sharex=True)
    fig.suptitle(title, fontsize=11)
    for i, ax in enumerate(axes.flat):
        ax.plot(X_sample[:, i], lw=1.2, color="steelblue")
        ax.set_title(CHANNEL_NAMES[i], fontsize=8)
        ax.grid(alpha=0.2)
    plt.tight_layout()
    return fig


def fig_confidence_bar(probs: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 3))
    colours = [ACTIVITY_COLORS[l] for l in LABEL_NAMES]
    bars    = ax.barh(LABEL_NAMES, probs, color=colours, edgecolor="white")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability")
    ax.set_title("Class Probabilities", fontweight="bold")
    for bar, p in zip(bars, probs):
        ax.text(min(p + 0.01, 0.99), bar.get_y() + bar.get_height() / 2,
                f"{p:.3f}", va="center", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    return fig


def fig_training_history(history: dict, model_name: str) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history["loss"],     label="Train")
    ax1.plot(history["val_loss"], label="Val", ls="--")
    ax1.set_title(f"{model_name} — Loss")
    ax1.set_xlabel("Epoch"); ax1.legend(); ax1.grid(alpha=0.25)

    ax2.plot(history["accuracy"],     label="Train")
    ax2.plot(history["val_accuracy"], label="Val", ls="--")
    ax2.set_title(f"{model_name} — Accuracy")
    ax2.set_xlabel("Epoch"); ax2.legend(); ax2.grid(alpha=0.25)
    plt.tight_layout()
    return fig


def fig_attention_single(attn_weights: np.ndarray, signal: np.ndarray) -> plt.Figure:
    T_orig = len(signal)
    T_attn = len(attn_weights)
    attn_up = np.interp(
        np.linspace(0, T_attn - 1, T_orig),
        np.arange(T_attn), attn_weights,
    )
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    ax1.plot(signal, lw=1.5, color="steelblue", label="Signal (ch 0)")
    ax1.fill_between(range(T_orig), signal.min(),
                     signal.min() + (signal.max() - signal.min()) * attn_up,
                     alpha=0.4, color="orange", label="Attention")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.2)
    ax1.set_title("Input Signal + Attention Overlay")

    ax2.bar(range(T_attn), attn_weights, color="orange", edgecolor="white")
    ax2.set_title("Raw Attention Weights (per compressed timestep)")
    ax2.set_xlabel("Compressed Timestep")
    ax2.set_ylabel("Weight")
    ax2.grid(alpha=0.2)
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  App layout
# ══════════════════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(
        page_title="HAR Real-Time System",
        page_icon="🏃",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("🏃 HAR System")
        st.markdown("**Real-Time Human Activity Recognition**")
        st.divider()

        page = st.radio(
            "Navigation",
            ["🏠 Home", "📡 Live Prediction", "📊 Model Comparison",
             "🔬 Explainability", "📈 Training History", "🧪 Robustness Lab"],
        )

        st.divider()
        model_choice = st.selectbox(
            "Model",
            ["CNN_LSTM_Attention", "CNN_LSTM", "LSTM", "ANN"],
        )
        model = load_model(model_choice)
        if model is None:
            st.error(f"⚠️ {model_choice}.keras not found.\nRun train.py first.")

        noise_sigma = st.slider("Noise σ (robustness)", 0.0, 0.3, 0.0, 0.01)

    X_test, y_test = load_test_data()

    # ─────────────────────────────────────────────────────────────────────────
    if "🏠 Home" in page:
        _page_home(model_choice)

    elif "📡 Live Prediction" in page:
        _page_live(model, model_choice, X_test, y_test, noise_sigma)

    elif "📊 Model Comparison" in page:
        _page_comparison()

    elif "🔬 Explainability" in page:
        _page_explainability(model, model_choice, X_test, y_test)

    elif "📈 Training History" in page:
        _page_history(model_choice)

    elif "🧪 Robustness Lab" in page:
        _page_robustness(model, model_choice, X_test, y_test)


# ══════════════════════════════════════════════════════════════════════════════
#  Pages
# ══════════════════════════════════════════════════════════════════════════════

def _page_home(model_choice):
    st.title("🏃 Real-Time Human Activity Recognition")
    st.markdown("""
    ### CNN-LSTM with Custom Attention Mechanism

    This system classifies **6 daily activities** from wrist/waist accelerometer
    and gyroscope data collected by smartphones.

    | Activity | Description |
    |---|---|
    | 🚶 WALKING | Level ground walking |
    | 🏃⬆️ WALKING_UPSTAIRS | Climbing stairs |
    | 🏃⬇️ WALKING_DOWNSTAIRS | Descending stairs |
    | 🪑 SITTING | Seated stationary |
    | 🧍 STANDING | Upright stationary |
    | 🛏️ LAYING | Horizontal/resting |

    ---
    ### Architecture
    ```
    Input (128 timesteps × 9 sensor channels)
      → Conv1D(64) → BatchNorm → MaxPool
      → Conv1D(128) → BatchNorm → MaxPool
      → LSTM(100, return_sequences=True)
      → Custom Attention Layer
      → Dense(64) → Dropout
      → Softmax Output (6 classes)
    ```
    ---
    ### Dataset
    - **UCI HAR Dataset** — 10,299 labelled windows
    - 30 subjects, ages 19–48
    - Samsung Galaxy S II @ 50 Hz
    """)

    col1, col2, col3 = st.columns(3)
    col1.metric("Window Size",   "128 samples")
    col2.metric("Overlap",       "50%")
    col3.metric("Sensor Channels", "9")


def _page_live(model, model_name, X_test, y_test, noise_sigma):
    st.title("📡 Live Prediction")

    if model is None or X_test is None:
        st.error("Load a model and data first.")
        return

    # ── Input mode ────────────────────────────────────────────────────────────
    input_mode = st.radio("Input Source", ["📂 Use Test Data", "📤 Upload CSV"])
    st.divider()

    if input_mode == "📂 Use Test Data":
        idx = st.slider("Sample index", 0, len(X_test) - 1, 0)
        X_sample = X_test[idx]
        true_label = LABEL_NAMES[y_test[idx]]
    else:
        f = st.file_uploader(
            "Upload sensor CSV (9 columns × 128 rows)", type=["csv"]
        )
        if f is None:
            st.info("Upload a CSV with 9 sensor channels, 128 rows per window.")
            st.code("Columns: " + ", ".join(CHANNEL_NAMES))
            return
        try:
            df_upload = pd.read_csv(f, header=None)
            X_sample  = df_upload.values[:128, :9].astype(np.float32)
            true_label = "Unknown"
        except Exception as e:
            st.error(f"CSV parse error: {e}")
            return

    # ── Apply noise ───────────────────────────────────────────────────────────
    X_input = X_sample.copy()
    if noise_sigma > 0:
        X_input += np.random.normal(0, noise_sigma, X_input.shape)
        st.info(f"🔊 Gaussian noise applied (σ={noise_sigma:.2f})")

    # ── Predict ───────────────────────────────────────────────────────────────
    t0   = time.perf_counter()
    prob = model.predict(X_input[np.newaxis, ...], verbose=0)[0]
    lat  = (time.perf_counter() - t0) * 1000

    pred_idx   = int(np.argmax(prob))
    pred_label = LABEL_NAMES[pred_idx]
    conf       = float(prob[pred_idx])

    # ── Display ───────────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Predicted Activity",
                f"{ACTIVITY_EMOJIS[pred_label]} {pred_label}")
    col2.metric("Confidence", f"{conf*100:.1f}%")
    col3.metric("True Label",  true_label)
    col4.metric("Latency",    f"{lat:.1f} ms")

    correct = (pred_label == true_label)
    if true_label != "Unknown":
        if correct:
            st.success("✅ Correct prediction!")
        else:
            st.error(f"❌ Incorrect — True: {true_label}")

    st.divider()

    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("Sensor Signals")
        st.pyplot(fig_signal(X_input, "Input Window (128 timesteps)"))
    with col_r:
        st.subheader("Class Probabilities")
        st.pyplot(fig_confidence_bar(prob))

    # ── Simulate stream ───────────────────────────────────────────────────────
    st.divider()
    st.subheader("🌊 Stream Simulation")
    n_stream = st.slider("Number of windows to stream", 50, 500, 200, 50)
    if st.button("▶ Run Stream"):
        predictor = RealTimePredictor(model, window_size=128, step_size=16)
        df = predictor.predict_stream(
            X_test[:n_stream],
            add_noise=(noise_sigma > 0),
            noise_sigma=noise_sigma,
        )

        fig2, ax = plt.subplots(figsize=(14, 3))
        colour_map = {l: i for i, l in enumerate(LABEL_NAMES)}
        for _, row in df.iterrows():
            c = plt.cm.tab10(colour_map[row["pred_label"]] / 10)
            ax.scatter(row["sample_idx"], colour_map[row["pred_label"]],
                       color=c, s=15)
        ax.set_yticks(range(len(LABEL_NAMES)))
        ax.set_yticklabels(LABEL_NAMES, fontsize=8)
        ax.set_xlabel("Sample")
        ax.set_title("Predicted Activity Timeline")
        ax.grid(alpha=0.2)
        plt.tight_layout()
        st.pyplot(fig2)

        fig3, ax3 = plt.subplots(figsize=(14, 2.5))
        ax3.fill_between(df["sample_idx"], df["confidence"], alpha=0.4)
        ax3.plot(df["sample_idx"], df["confidence"], lw=1.5)
        ax3.set_ylim(0, 1.05)
        ax3.set_ylabel("Confidence"); ax3.set_xlabel("Sample")
        ax3.set_title("Confidence Over Time"); ax3.grid(alpha=0.2)
        plt.tight_layout()
        st.pyplot(fig3)

        st.dataframe(df[["sample_idx","pred_label","confidence"]].head(20))
        csv_bytes = df.to_csv(index=False).encode()
        st.download_button("⬇ Download Predictions CSV", csv_bytes,
                           "predictions.csv", "text/csv")


def _page_comparison():
    st.title("📊 Model Comparison")
    df = load_comparison_csv()
    if df is None:
        st.warning("Run `python src/evaluate.py` to generate comparison data.")
        st.info("Expected file: `outputs/model_comparison.csv`")
        return

    st.dataframe(df.style.highlight_max(
        subset=["Accuracy","Precision","Recall","F1"],
        color="#c8f7c5", axis=0,
    ), use_container_width=True)

    # ── Chart ─────────────────────────────────────────────────────────────────
    metrics = ["Accuracy","Precision","Recall","F1"]
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(df))
    w = 0.2
    for i, m in enumerate(metrics):
        ax.bar(x + i * w, df[m], width=w, label=m)
    ax.set_xticks(x + w * 1.5)
    ax.set_xticklabels(df["Model"], rotation=12, ha="right")
    ax.set_ylim(0.5, 1.02)
    ax.set_title("Model Comparison — All Metrics", fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    st.pyplot(fig)

    # ── Robustness comparison ─────────────────────────────────────────────────
    st.subheader("Robustness Drop (σ = 0.10 vs clean)")
    rows = []
    for mname in ["ANN","LSTM","CNN_LSTM","CNN_LSTM_Attention"]:
        p = ROOT / "outputs" / f"{mname}_robustness.csv"
        if p.exists():
            rob = pd.read_csv(p)
            clean = rob.loc[rob["Noise σ"] == 0.0, "Accuracy"].values
            noisy = rob.loc[rob["Noise σ"] == 0.1,  "Accuracy"].values
            if len(clean) and len(noisy):
                rows.append({
                    "Model": mname,
                    "Clean Acc": round(float(clean[0]), 4),
                    "Noisy Acc (σ=0.1)": round(float(noisy[0]), 4),
                    "Drop": round(float(clean[0] - noisy[0]), 4),
                })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True)


def _page_explainability(model, model_name, X_test, y_test):
    st.title("🔬 Model Explainability")

    if model is None or X_test is None:
        st.error("Load a model and data first.")
        return

    if model_name != "CNN_LSTM_Attention":
        st.warning("Attention visualisation is only available for CNN_LSTM_Attention.")
        return

    idx = st.slider("Sample index", 0, len(X_test) - 1, 42)
    X_sample   = X_test[idx : idx + 1]
    true_label = LABEL_NAMES[y_test[idx]]

    try:
        attn = get_attention_weights(model, X_sample)[0]   # (T',)
    except Exception as e:
        st.error(f"Could not extract attention weights: {e}")
        return

    prob       = model.predict(X_sample, verbose=0)[0]
    pred_label = LABEL_NAMES[np.argmax(prob)]

    st.markdown(f"**True:** `{true_label}` &nbsp;&nbsp; **Predicted:** `{pred_label}` "
                f"&nbsp;&nbsp; **Conf:** `{prob.max():.3f}`")
    st.divider()

    st.subheader("Attention Overlay")
    st.pyplot(fig_attention_single(attn, X_test[idx, :, 0]))

    st.subheader("What the attention weights mean")
    st.markdown("""
    - **Orange overlay**: Timesteps the model focuses on most.
    - **Bar chart**: Raw attention distribution over compressed time axis.
    - Higher bars → model relies more on those time frames for classification.
    - For *dynamic* activities (walking, stairs) attention tends to spread evenly.
    - For *static* activities (sitting, standing, laying) attention concentrates
      on fewer, discriminative frames.
    """)

    # Saved static plots
    st.subheader("Saved Attention Maps")
    col1, col2 = st.columns(2)
    for p, col in [
        (PLOTS_DIR / "attention_heatmaps.png",   col1),
        (PLOTS_DIR / "attention_per_class.png",  col2),
    ]:
        if p.exists():
            col.image(str(p), use_column_width=True)
        else:
            col.info(f"Run `python src/explainability.py` to generate: {p.name}")

    # SHAP
    st.subheader("SHAP Channel Importance")
    shap_p = PLOTS_DIR / "shap_channel_importance.png"
    if shap_p.exists():
        st.image(str(shap_p), use_column_width=True)
    else:
        st.info("Run `python src/explainability.py` to generate SHAP plot.")


def _page_history(model_name):
    st.title("📈 Training History")
    history = load_history(model_name)
    if history is None:
        st.warning(f"No training history for {model_name}. Run train.py first.")
        return
    st.pyplot(fig_training_history(history, model_name))

    # All models side by side
    st.subheader("All Models — Final Val Accuracy")
    rows = []
    for name in ["ANN","LSTM","CNN_LSTM","CNN_LSTM_Attention"]:
        h = load_history(name)
        if h:
            rows.append({
                "Model":    name,
                "Best Val Acc":  round(max(h["val_accuracy"]), 4),
                "Final Val Acc": round(h["val_accuracy"][-1], 4),
                "Epochs":        len(h["loss"]),
            })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True)


def _page_robustness(model, model_name, X_test, y_test):
    st.title("🧪 Robustness Lab")

    if model is None or X_test is None:
        st.error("Load a model and data first.")
        return

    p = ROOT / "outputs" / f"{model_name}_robustness.csv"
    if p.exists():
        df = pd.read_csv(p)
        st.subheader(f"Pre-computed: {model_name}")
        st.dataframe(df, use_container_width=True)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(df["Noise σ"], df["Accuracy"], "o-", lw=2, label="Accuracy")
        ax.plot(df["Noise σ"], df["F1"],       "s--",lw=2, label="F1")
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Noise σ"); ax.set_ylabel("Score")
        ax.set_title(f"{model_name} — Robustness Under Noise", fontweight="bold")
        ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("Run `python src/evaluate.py` to pre-compute robustness data.")

    st.divider()
    st.subheader("Interactive Noise Test")
    sigma = st.slider("Choose noise σ", 0.0, 0.30, 0.05, 0.01)
    n_test = st.slider("Samples to test", 100, 1000, 300, 100)

    if st.button("🧪 Run Test"):
        X_noisy = HARDataPipeline.inject_noise(X_test[:n_test], sigma=sigma)
        y_pred  = np.argmax(model.predict(X_noisy, verbose=0), axis=1)
        from sklearn.metrics import accuracy_score, f1_score
        acc = accuracy_score(y_test[:n_test], y_pred)
        f1  = f1_score(y_test[:n_test], y_pred, average="weighted", zero_division=0)
        c1, c2 = st.columns(2)
        c1.metric("Accuracy", f"{acc:.4f}")
        c2.metric("F1 Score", f"{f1:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Also expose PLOTS_DIR, OUTPUTS_DIR to page helpers
    PLOTS_DIR   = ROOT / "plots"
    OUTPUTS_DIR = ROOT / "outputs"
    main()
