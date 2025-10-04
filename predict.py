# predict.py
import os
import joblib
import torch
import numpy as np
import pandas as pd

# ✅ Only import the model class
from train_embedding import CarModel  

# -------------------------------
# Paths
# -------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "deep_model.pth")
PREP_PATH = os.path.join(PROJECT_ROOT, "models", "preprocessed.pkl")

if __name__ == "__main__":
    # -------------------------------
    # Load preprocessing artifacts
    # -------------------------------
    preprocessed = joblib.load(PREP_PATH)
    encoders = preprocessed["encoders"]
    onehot_enc = preprocessed["onehot_enc"]
    scaler = preprocessed["scaler"]

    embed_cols = preprocessed["embed_cols"]
    onehot_cols = preprocessed["onehot_cols"]
    num_cols = preprocessed["num_cols"]
    onehot_features = preprocessed["onehot_feature_names"]


    # -------------------------------
    # Load raw dataset
    # -------------------------------
    DATA_PATH = os.path.join(PROJECT_ROOT, "data", "used_cars", "used_cars_data_clean.csv")
    df = pd.read_csv(DATA_PATH)

    # Target: log(price)
    y = np.log1p(df["price"].values)

    # --- FIX: clean numeric features BEFORE building arrays ---
    if "price" in num_cols:
        num_cols.remove("price")


    # -------------------------------
    # Define model (must match training!)
    # -------------------------------
    embed_sizes = [
        (len(encoders["make_name"].classes_), 50),
        (len(encoders["model_name"].classes_), 100),
    ]

    # # ✅ Compute input_size exactly like in training
    # dummy = pd.DataFrame([{c: onehot_enc.categories_[i][0] for i, c in enumerate(onehot_cols)}])
    # onehot_dim = onehot_enc.transform(dummy).shape[1]
    # input_size = len(num_cols) + onehot_dim

    # model = CarModel(embed_sizes, input_size)
    # model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    # model.eval()
    # print("✅ Model loaded successfully!")

    # Define model (must match training!)
    embed_sizes = [
        (len(encoders["make_name"].classes_), 50),
        (len(encoders["model_name"].classes_), 100),
    ]

    # ✅ Use saved input_size from training
    input_size = preprocessed["input_size"]

    model = CarModel(embed_sizes, input_size)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    print("✅ Model loaded successfully!")

    # -------------------------------
    # Default template
    # -------------------------------
    # 
    default_car = {
        "make_name": "Unknown",
        "model_name": "Unknown",
        "year": 2020,
        "mileage": 0,
        "owner_count": 1,   # only here once
        "daysonmarket": 0,
        "fuel_type": "Gasoline",
        "transmission_display": "Automatic",
        "trim_name": "Base",
        "city": "Unknown",
        "engine_type": "I4",
        "engine_cylinders": "4",
        "wheel_system_display": "FWD",
        "exterior_color": "Silver",
        "length": 192,
        "width": 72,
        "wheelbase": 111,
        "maximum_seating": 5,
        "theft_title": 0,
        "salvage": 0,
        "frame_damaged": 0,
        "has_accidents": 0,
        "isCab": 0,
        "fleet": 0,
        "listing_color": "Silver",
        "listed_year": 2023,
        "listed_month": 5,
        "franchise_dealer": 1, 
        "is_new": 0, 
        "savings_amount": 0,
        "sp_id": 0, 

        # body type (Camry = Sedan)
        "body_type_Convertible": 0, 
        "body_type_Coupe": 0,
        "body_type_Hatchback": 0,
        "body_type_Minivan": 0,
        "body_type_Pickup Truck": 0,
        "body_type_SUV / Crossover": 0,
        "body_type_Sedan": 1,
        "body_type_Unknown": 0,
        "body_type_Van": 0,
        "body_type_Wagon": 0,

        # engine type (Camry Hybrid = I4 Hybrid)
        "engine_type_H4": 0,
        "engine_type_H4 Hybrid": 0,
        "engine_type_H6": 0,
        "engine_type_I2": 0,
        "engine_type_I3": 0,
        "engine_type_I4": 0,
        "engine_type_I4 Diesel": 0,
        "engine_type_I4 Flex Fuel Vehicle": 0,
        "engine_type_I4 Hybrid": 1,   # ✅ Camry Hybrid
        "engine_type_I5": 0,
        "engine_type_I5 Biodiesel": 0,
        "engine_type_I6": 0,
        "engine_type_I6 Diesel": 0,
        "engine_type_R2": 0,
        "engine_type_Unknown": 0,
        "engine_type_V10": 0,
        "engine_type_V12": 0,
        "engine_type_V6": 0,
        "engine_type_V6 Biodiesel": 0,
        "engine_type_V6 Diesel": 0,
        "engine_type_V6 Flex Fuel Vehicle": 0,
        "engine_type_V6 Hybrid": 0,
        "engine_type_V8": 0,
        "engine_type_V8 Biodiesel": 0,
        "engine_type_V8 Compressed Natural Gas": 0,
        "engine_type_V8 Diesel": 0,
        "engine_type_V8 Flex Fuel Vehicle": 0,
        "engine_type_V8 Hybrid": 0,
        "engine_type_W12": 0,
        "engine_type_W12 Flex Fuel Vehicle": 0,

        # theft title flags
        "theft_title": 0,
        "theft_title_False": 1, 
        "theft_title_True": 0,
        "theft_title_Unknown": 0
    }

    new_car = default_car.copy()
    new_car.update({
        "make_name": "Toyota",
        "model_name": "Camry",
        "year": 2018,
        "mileage": 62000,
        "owner_count": 2,   # ✅ overrides the default
        "daysonmarket": 18,
        "fuel_type": "Hybrid",
        "transmission_display": "Automatic",
        "trim_name": "SE Hybrid",
        "city": "Raleigh",
        "listed_year": 2023,
        "listed_month": 7,
        "transmission": "Automatic",
        "wheel_system": "FWD",
    })

    # Convert dict → DataFrame
    sample = pd.DataFrame([new_car])

    # ✅ Normalize ALL string columns (case-insensitive, trimmed)
    for col in sample.select_dtypes(include=["object", "category"]).columns:
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
    # One-hot + numeric
    # -------------------------------
    print("DEBUG: num_cols list from preprocessed.pkl")
    print(num_cols)
    print("\nDEBUG: sample[num_cols] values (first row):")
    print(sample[num_cols].iloc[0].to_dict())

    onehot_input = onehot_enc.transform(sample[onehot_cols])  # numpy array

    # ✅ only scale numeric cols (make sure no strings are inside)
    try:
        num_input = scaler.transform(sample[num_cols])            # scaled numeric
    except Exception as e:
        print("❌ ERROR during scaling:", e)
        print("Offending values:\n", sample[num_cols].dtypes)
        raise

    num_input = scaler.transform(sample[num_cols])            # scaled numeric
    other_input = np.hstack([num_input, onehot_input])
    other_input = torch.tensor(other_input, dtype=torch.float32)

    print("DEBUG (PREDICT):")
    print("num_input shape:", num_input.shape)
    print("onehot_input shape:", onehot_input.shape)
    print("other_input shape:", other_input.shape)

    print("Scaled num_input (first row):", num_input[0][:10])  # first 10 numeric features
    print("Onehot nonzero count:", np.count_nonzero(onehot_input))
    print("Embed indices:", embed_input.tolist())

    print("make_name classes:", encoders["make_name"].classes_[:20])  # first 20
    print("model_name classes:", encoders["model_name"].classes_[:20])



    # -------------------------------
    # Predict (reverse log transform)
    # -------------------------------
    with torch.no_grad():
        pred_log = model(embed_input, other_input).item()
        pred_price = np.expm1(pred_log)

    print(f"💰 Predicted Price: ${pred_price:,.2f}")
    print(f"📉 Predicted Log Price: {pred_log:.2f}")
