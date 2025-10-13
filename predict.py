"""
Prediction script for used-car price estimation model.

Loads the trained deep learning model and preprocessing pipeline,
then predicts price for a fully aligned feature sample (CSV input).
"""

import os
import joblib
import torch
import numpy as np
import pandas as pd
from train_embedding import CarModel

# -------------------------------
# Paths
# -------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "deep_model.pth")
PREP_PATH = os.path.join(PROJECT_ROOT, "models", "preprocessed.pkl")
SAMPLE_PATH = os.path.join(PROJECT_ROOT, "sample_car_input.csv")

# -------------------------------
# Load preprocessing + model
# -------------------------------
preprocessed = joblib.load(PREP_PATH)
encoders = preprocessed["encoders"]
onehot_enc = preprocessed["onehot_enc"]
scaler = preprocessed["scaler"]
embed_cols = preprocessed["embed_cols"]
num_cols = preprocessed["num_cols"]
onehot_cols = preprocessed["onehot_cols"]
input_size = preprocessed["input_size"]

embed_sizes = [
    (len(encoders["make_name"].classes_), 50),
    (len(encoders["model_name"].classes_), 100),
]

model = CarModel(embed_sizes, input_size)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()
print("✅ Model loaded successfully!")

# -------------------------------
# Load aligned sample input
# -------------------------------
sample = pd.read_csv(SAMPLE_PATH, index_col=0).T
sample.reset_index(drop=True, inplace=True)

# Normalize string columns
for col in sample.select_dtypes(include=["object"]).columns:
    sample[col] = sample[col].astype(str).str.strip().str.lower()

# -------------------------------
# Encode embeddings
# -------------------------------
embed_input = []
for col in embed_cols:
    val = sample[col].values[0]
    idx = np.where(encoders[col].classes_ == val)[0][0] if val in encoders[col].classes_ else 0
    embed_input.append(idx)
embed_input = torch.tensor([embed_input], dtype=torch.long)

# -------------------------------
# Numeric + one-hot features
# -------------------------------
# Ensure all numeric columns exist
for c in num_cols:
    if c not in sample.columns:
        sample[c] = 0.0

# One-hot columns — skip, since already aligned in sample_car_input.csv
X_num = scaler.transform(sample[num_cols]).astype(np.float32)

# Drop embedding columns
other_input = sample.drop(columns=embed_cols, errors="ignore").values.astype(np.float32)
other_input = torch.tensor(other_input, dtype=torch.float32)

# -------------------------------
# Sanity check
# -------------------------------
print(f"✅ Feature alignment check:\nExpected input size: {input_size}\nGot: {other_input.shape[1]}")

# -------------------------------
# Predict (reverse log transform)
# -------------------------------
with torch.no_grad():
    pred_log = model(embed_input, other_input).item()
    pred_price = np.expm1(pred_log)

print(f"\n💰 Predicted Price: ${pred_price:,.2f}")
print(f"📉 Predicted Log Price: {pred_log:.2f}")
