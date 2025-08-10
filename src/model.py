from __future__ import annotations
from tensorflow import keras
from tensorflow.keras import layers

def build_model(input_dim: int,
                learning_rate: float = 1e-3,
                hidden1: int = 64,
                hidden2: int = 128,
                dropout: float = 0.2) -> keras.Model:
    """MLP for tabular regression: sigmoid -> ReLU -> Dropout -> linear."""
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(hidden1, activation="sigmoid"),
        layers.Dense(hidden2, activation="relu"),
        layers.Dropout(dropout),
        layers.Dense(1, activation="linear"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mae",
        metrics=["mae", "mse"],
    )
    return model
