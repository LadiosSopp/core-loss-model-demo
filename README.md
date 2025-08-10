# Core Loss Prediction & Furnace Setpoint Optimization (De-identified Demo)

This repository demonstrates a **production-style** implementation for predicting electrical steel *core loss* from process/material features and using that estimate to **suggest optimized furnace temperature setpoints**. All company/product identifiers were removed; data and endpoints are synthetic.

> Built with TensorFlow/Keras and standard Python tooling. Ready for local training, evaluation, and exporting a light-weight prediction CLI or service.

## Why this matters
Reducing core loss while meeting magnetic performance targets improves motor efficiency and energy footprint. A reliable ML predictor enables **lower, safer setpoints** and **dynamic adjustments** without sacrificing quality.

## Highlights
- Clean training pipeline with normalization artifacts persisted to disk.
- Keras MLP: `sigmoid → ReLU → Dropout → linear` (fast, stable for tabular regression).
- CLI tools for train/eval/predict; optional FastAPI service stub.
- Fully de-identified; includes **synthetic sample data** and plots for LinkedIn.

## Quickstart
```bash
python -m venv .venv && . .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Train on synthetic sample (replace with your CSV)
python src/train.py --train_csv examples/data/synthetic_train.csv --target_column core_loss   --epochs 60 --model_path assets/model.h5 --normalizer_path assets/normalizer.npz

# Evaluate
python src/evaluate.py --data_csv examples/data/synthetic_test.csv --target_column core_loss   --model_path assets/model.h5 --normalizer_path assets/normalizer.npz

# Batch predict
python src/predict_cli.py --input_csv examples/data/synthetic_test.csv   --output_csv examples/data/predicted.csv --model_path assets/model.h5   --normalizer_path assets/normalizer.npz
```

## Data schema (example)
Your CSV should contain **feature columns** (e.g., `f1..fN` or domain features) and a numeric target column (default: `core_loss`). Missing values are imputed to zero by default; customize in `src/data.py`.

## Notes on model
- Hidden stack mixes **sigmoid** (to capture signed feature interactions) and **ReLU** (sparse activations) with **Dropout** for regularization.
- Loss: MAE; Metrics: MAE/MSE and R².
- See `src/model.py` for hyperparameters.

## Folder layout
```
assets/               # Saved model + normalizer artifacts
examples/data/        # Synthetic CSVs you can run immediately
figures/              # Training and evaluation plots (for LinkedIn)
src/                  # Library and CLIs
  data.py
  model.py
  train.py
  evaluate.py
  predict_cli.py
  service_stub.py     # Optional FastAPI service (de-identified)
```

## License
MIT (see `LICENSE`).

---

*Maintainer:* @LadiosSopp  
*Disclaimer:* This demo is de-identified and ships with synthetic data; replace with your own features/target before production use.
