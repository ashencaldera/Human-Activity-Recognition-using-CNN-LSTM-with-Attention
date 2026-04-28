"""
src/models.py
─────────────
Four model architectures for progressive comparison.

  Model A  –  Baseline ANN       (Dense only)
  Model B  –  LSTM               (Recurrent only)
  Model C  –  CNN + LSTM         (Hybrid)
  Model D  –  CNN + LSTM + Attn  (Final / Best)

Custom Attention layer is defined here and exported separately
for use in explainability scripts.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
#  Custom Attention Layer
# ══════════════════════════════════════════════════════════════════════════════

class AttentionLayer(tf.keras.layers.Layer):
    """
    Additive (Bahdanau-style) self-attention over the time dimension.

    Input  : (batch, timesteps, features)
    Output : (batch, features)   — context vector

    The learnable weight matrix W maps each timestep feature vector to a
    scalar score. Softmax over time gives the attention distribution;
    the weighted sum collapses the time axis to a single context vector.

    Attributes
    ----------
    attention_weights : np.ndarray
        Stored after the last forward pass; shape (batch, timesteps, 1).
        Used by the explainability module.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.attention_weights = None

    def build(self, input_shape):
        # W : (feature_dim, 1)
        self.W = self.add_weight(
            name="attn_W",
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.b = self.add_weight(
            name="attn_b",
            shape=(1,),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x, return_weights=False):
        # score : (batch, timesteps, 1)
        score = tf.nn.tanh(tf.matmul(x, self.W) + self.b)
        # weights : (batch, timesteps, 1)  — soft alignment over time
        weights = tf.nn.softmax(score, axis=1)
        self.attention_weights = weights          # cache for explainability
        # weighted sum → context : (batch, features)
        context = tf.reduce_sum(weights * x, axis=1)
        if return_weights:
            return context, weights
        return context

    def get_config(self):
        return super().get_config()


# ══════════════════════════════════════════════════════════════════════════════
#  Helper – compile settings
# ══════════════════════════════════════════════════════════════════════════════

def _compile(model, lr=1e-3):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ══════════════════════════════════════════════════════════════════════════════
#  Model A  –  Baseline ANN
# ══════════════════════════════════════════════════════════════════════════════

def build_ann(input_shape=(128, 9), n_classes=6) -> tf.keras.Model:
    """
    Fully-connected baseline.
    Flattens the (128×9) window and feeds through Dense layers.
    """
    inp = layers.Input(shape=input_shape, name="input")
    x   = layers.Flatten()(inp)
    x   = layers.Dense(256, activation="relu",
                        kernel_regularizer=regularizers.l2(1e-4))(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dropout(0.4)(x)
    x   = layers.Dense(128, activation="relu",
                        kernel_regularizer=regularizers.l2(1e-4))(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dropout(0.3)(x)
    x   = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(n_classes, activation="softmax", name="output")(x)

    model = models.Model(inp, out, name="ANN_Baseline")
    return _compile(model)


# ══════════════════════════════════════════════════════════════════════════════
#  Model B  –  LSTM
# ══════════════════════════════════════════════════════════════════════════════

def build_lstm(input_shape=(128, 9), n_classes=6) -> tf.keras.Model:
    """
    Stacked LSTM for sequence modelling.
    """
    inp = layers.Input(shape=input_shape, name="input")
    x   = layers.LSTM(128, return_sequences=True, name="lstm_1")(inp)
    x   = layers.Dropout(0.3)(x)
    x   = layers.LSTM(64, return_sequences=False, name="lstm_2")(x)
    x   = layers.Dropout(0.3)(x)
    x   = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(n_classes, activation="softmax", name="output")(x)

    model = models.Model(inp, out, name="LSTM")
    return _compile(model)


# ══════════════════════════════════════════════════════════════════════════════
#  Model C  –  CNN + LSTM
# ══════════════════════════════════════════════════════════════════════════════

def build_cnn_lstm(input_shape=(128, 9), n_classes=6) -> tf.keras.Model:
    """
    CNN extracts local temporal features; LSTM captures long-range dependencies.
    """
    inp = layers.Input(shape=input_shape, name="input")

    # ── CNN block 1 ──────────────────────────────────────────────────────
    x = layers.Conv1D(64, kernel_size=3, padding="same", activation="relu",
                      name="conv1")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.2)(x)

    # ── CNN block 2 ──────────────────────────────────────────────────────
    x = layers.Conv1D(128, kernel_size=3, padding="same", activation="relu",
                      name="conv2")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.2)(x)

    # ── LSTM ─────────────────────────────────────────────────────────────
    x = layers.LSTM(100, return_sequences=False, name="lstm")(x)
    x = layers.Dropout(0.3)(x)

    # ── Head ─────────────────────────────────────────────────────────────
    x   = layers.Dense(64, activation="relu")(x)
    x   = layers.Dropout(0.3)(x)
    out = layers.Dense(n_classes, activation="softmax", name="output")(x)

    model = models.Model(inp, out, name="CNN_LSTM")
    return _compile(model)


# ══════════════════════════════════════════════════════════════════════════════
#  Model D  –  CNN + LSTM + Attention  (FINAL MODEL)
# ══════════════════════════════════════════════════════════════════════════════

def build_cnn_lstm_attention(input_shape=(128, 9), n_classes=6) -> tf.keras.Model:
    """
    Full pipeline:
      Conv1D → BN → Pool
      Conv1D → BN → Pool
      LSTM (return_sequences=True)
      Custom Attention layer
      Dense head
    """
    inp = layers.Input(shape=input_shape, name="input")

    # ── CNN block 1 ──────────────────────────────────────────────────────
    x = layers.Conv1D(64, kernel_size=3, padding="same", activation="relu",
                      name="conv1")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.2)(x)

    # ── CNN block 2 ──────────────────────────────────────────────────────
    x = layers.Conv1D(128, kernel_size=3, padding="same", activation="relu",
                      name="conv2")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.2)(x)

    # ── LSTM with sequence output (needed for Attention) ──────────────────
    x = layers.LSTM(100, return_sequences=True, name="lstm")(x)

    # ── Custom Attention ──────────────────────────────────────────────────
    x = AttentionLayer(name="attention")(x)     # → (batch, 100)

    # ── Classification head ───────────────────────────────────────────────
    x   = layers.Dense(64, activation="relu",
                        kernel_regularizer=regularizers.l2(1e-4))(x)
    x   = layers.Dropout(0.4)(x)
    out = layers.Dense(n_classes, activation="softmax", name="output")(x)

    model = models.Model(inp, out, name="CNN_LSTM_Attention")
    return _compile(model)


# ══════════════════════════════════════════════════════════════════════════════
#  Factory
# ══════════════════════════════════════════════════════════════════════════════

MODEL_REGISTRY = {
    "ANN":                build_ann,
    "LSTM":               build_lstm,
    "CNN_LSTM":           build_cnn_lstm,
    "CNN_LSTM_Attention": build_cnn_lstm_attention,
}


def build_model(name: str, input_shape=(128, 9), n_classes=6) -> tf.keras.Model:
    """
    Factory function.

    Parameters
    ----------
    name : str
        One of 'ANN', 'LSTM', 'CNN_LSTM', 'CNN_LSTM_Attention'.

    Returns
    -------
    Compiled tf.keras.Model
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](input_shape=input_shape, n_classes=n_classes)


# ─── Quick summary ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for name in MODEL_REGISTRY:
        m = build_model(name)
        print(f"\n{'='*60}")
        m.summary()
