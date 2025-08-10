# Optional FastAPI stub for serving predictions (de-identified; no real endpoints).
from __future__ import annotations
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np
import tensorflow as tf
from data import load_normalizer, normalize_feat

app = FastAPI(title="Core Loss Predictor (Demo)")
_model = None
_mu = None
_sigma = None

class PredictRequest(BaseModel):
    features: List[float]  # length must match training feature count

def ensure_loaded(model_path="assets/model.h5", normalizer_path="assets/normalizer.npz"):
    global _model, _mu, _sigma
    if _model is None:
        _model = tf.keras.models.load_model(model_path)
        _mu, _sigma = load_normalizer(normalizer_path)

@app.post("/predict")
def predict(req: PredictRequest):
    ensure_loaded()
    X = np.array(req.features, dtype="float32").reshape(1, -1)
    Xn = normalize_feat(X, _mu, _sigma)
    y = _model.predict(Xn, verbose=0).flatten()[0]
    return {"core_loss_pred": float(y)}
