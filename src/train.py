from __future__ import annotations
import argparse, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from data import load_csv, compute_mean_std, normalize_feat, save_normalizer
from model import build_model

def plot_and_save(y_true, y_pred, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    # Pred vs True
    plt.figure()
    plt.scatter(y_true, y_pred, s=10)
    lims = [float(min(y_true.min(), y_pred.min())), float(max(y_true.max(), y_pred.max()))]
    plt.plot(lims, lims)
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    plt.title("Predicted vs True")
    plt.savefig(os.path.join(out_dir, "pred_vs_true.png"), bbox_inches="tight")
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--target_column", default="core_loss")
    ap.add_argument("--model_path", default="assets/model.h5")
    ap.add_argument("--normalizer_path", default="assets/normalizer.npz")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--val_split", type=float, default=0.2)
    ap.add_argument("--learning_rate", type=float, default=1e-3)
    args = ap.parse_args()

    X_df, y = load_csv(args.train_csv, target_column=args.target_column)
    X = X_df.values.astype("float32")
    mu, sigma = compute_mean_std(X)
    Xn = normalize_feat(X, mu, sigma)
    save_normalizer(args.normalizer_path, mu, sigma)

    model = build_model(Xn.shape[1], learning_rate=args.learning_rate)

    cbs = [
        EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", patience=10, factor=0.5),
        ModelCheckpoint(args.model_path, monitor="val_loss", save_best_only=True),
    ]

    hist = model.fit(
        Xn, y.values,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=args.val_split,
        verbose=1,
        callbacks=cbs,
    )

    # Evaluate quickly (training+holdout)
    y_pred = model.predict(Xn, verbose=0).flatten()
    from sklearn.metrics import r2_score
    r2 = r2_score(y, y_pred)
    print(f"Training+holdout R2: {r2:.4f}")

    plot_and_save(y, y_pred, out_dir="figures")

if __name__ == "__main__":
    main()
