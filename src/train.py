import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np

def train_model(data_path, model_out="preprocessed.pkl"):
    df = pd.read_csv(data_path)

    # -------------------------------
    # Target + Features
    # -------------------------------
    y = df["price"].values
    X = df.drop(columns=["price"])

    # -------------------------------
    # Define feature groups
    # -------------------------------
    embed_cols = ["make_name", "model_name"]
    onehot_cols = ["fuel_type", "transmission_display", "trim_name", "city"]
    num_cols = [
        "mileage", "owner_count", "daysonmarket", "year",
        "listed_year", "listed_month", "latitude", "longitude"
    ]

    # -------------------------------
    # Label encode embedding columns
    # -------------------------------
    encoders = {}
    for col in embed_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

    # -------------------------------
    # One-hot encode small categoricals
    # -------------------------------
    onehot_enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    onehot_data = onehot_enc.fit_transform(X[onehot_cols])
    onehot_feature_names = onehot_enc.get_feature_names_out(onehot_cols)

    # -------------------------------
    # Numeric
    # -------------------------------
    X_num = X[num_cols].values

    # -------------------------------
    # Final arrays
    # -------------------------------
    X_embed = X[embed_cols].values
    X_other = np.hstack([X_num, onehot_data])

    # -------------------------------
    # Train/val split
    # -------------------------------
    X_embed_train, X_embed_val, X_other_train, X_other_val, y_train, y_val = train_test_split(
        X_embed, X_other, y, test_size=0.2, random_state=42
    )

    # -------------------------------
    # Save encoders + splits
    # -------------------------------
    save_dict = {
        "X_embed_train": X_embed_train,
        "X_embed_val": X_embed_val,
        "X_other_train": X_other_train,
        "X_other_val": X_other_val,
        "y_train": y_train,
        "y_val": y_val,
        "encoders": encoders,
        "onehot_enc": onehot_enc,
        "onehot_feature_names": onehot_feature_names,
        "embed_cols": embed_cols,
        "onehot_cols": onehot_cols,
        "num_cols": num_cols,
    }

    joblib.dump(save_dict, model_out)
    print(f"✅ Preprocessed data + encoders saved to {model_out}")

    return save_dict
