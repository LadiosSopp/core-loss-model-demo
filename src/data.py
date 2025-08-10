from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple, Optional

EPS = 1e-8

def load_csv(path: str, target_column: Optional[str] = None) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """Load CSV and split features/target (optional)."""
    df = pd.read_csv(path)
    df = df.fillna(0)  # Simple impute; customize as needed
    if target_column is None or target_column not in df.columns:
        return df, None
    y = df[target_column].astype("float32")
    X = df.drop(columns=[target_column]).astype("float32")
    return X, y

def compute_mean_std(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return per-feature mean and std with safe floor to avoid division by zero."""
    mu = X.mean(axis=0, keepdims=True).astype("float32")
    sigma = X.std(axis=0, keepdims=True).astype("float32")
    sigma = np.maximum(sigma, EPS)
    return mu, sigma

def normalize_feat(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return (X - mu) / sigma

def save_normalizer(path: str, mu: np.ndarray, sigma: np.ndarray) -> None:
    np.savez(path, mu=mu, sigma=sigma)

def load_normalizer(path: str):
    data = np.load(path)
    return data["mu"], data["sigma"]
