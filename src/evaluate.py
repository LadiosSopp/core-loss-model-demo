from __future__ import annotations
import argparse, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import tensorflow as tf

from data import load_csv, load_normalizer, normalize_feat

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_csv", required=True)
    ap.add_argument("--target_column", default="core_loss")
    ap.add_argument("--model_path", default="assets/model.h5")
    ap.add_argument("--normalizer_path", default="assets/normalizer.npz")
    args = ap.parse_args()

    X_df, y = load_csv(args.data_csv, target_column=args.target_column)
    X = X_df.values.astype("float32")
    mu, sigma = load_normalizer(args.normalizer_path)
    Xn = normalize_feat(X, mu, sigma)

    model = tf.keras.models.load_model(args.model_path)
    y_pred = model.predict(Xn, verbose=0).flatten()

    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)

    print(f"R2={r2:.4f}  MAE={mae:.6f}  MSE={mse:.6f}")

    # Save a histogram of errors
    err = y_pred - y.values
    plt.figure()
    plt.hist(err, bins=25)
    plt.xlabel("Prediction Error")
    plt.ylabel("Count")
    plt.title("Error Distribution")
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/error_hist.png", bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    main()
