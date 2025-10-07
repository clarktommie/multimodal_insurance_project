"""End-to-end test for predict.py"""

import torch
import joblib
import numpy as np

def test_model_prediction():
    """Ensure model predicts a valid price"""
    preprocessed = joblib.load("models/preprocessed.pkl")
    model_path = "models/deep_model.pth"

    # Rebuild a lightweight model identical to train_embedding.py
    from train_embedding import CarModel

    embed_sizes = [
        (len(preprocessed["encoders"]["make_name"].classes_), 50),
        (len(preprocessed["encoders"]["model_name"].classes_), 100)
    ]

    model = CarModel(embed_sizes, preprocessed["input_size"])
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # Make dummy input (one sample)
    x_embed = torch.tensor([[10, 200]], dtype=torch.long)
    x_other = torch.randn((1, preprocessed["input_size"]), dtype=torch.float32)

    with torch.no_grad():
        pred = model(x_embed, x_other).item()

    assert isinstance(pred, float)
    assert 0 < pred < 200000, f"Prediction out of range: {pred}"
