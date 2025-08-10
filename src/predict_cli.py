from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

from data import load_csv, load_normalizer, normalize_feat

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--output_csv", required=True)
    ap.add_argument("--model_path", default="assets/model.h5")
    ap.add_argument("--normalizer_path", default="assets/normalizer.npz")
    ap.add_argument("--prediction_column", default="core_loss_pred")
    args = ap.parse_args()

    X_df, _ = load_csv(args.input_csv, target_column=None)
    mu, sigma = load_normalizer(args.normalizer_path)
    Xn = normalize_feat(X_df.values.astype("float32"), mu, sigma)

    model = tf.keras.models.load_model(args.model_path)
    y_pred = model.predict(Xn, verbose=0).flatten()

    out = X_df.copy()
    out[args.prediction_column] = y_pred
    out.to_csv(args.output_csv, index=False)
    print(f"Saved predictions to {args.output_csv}")

if __name__ == "__main__":
    main()
