from email import encoders
import os
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

def preprocess_data(data_path, out_path):
    df = pd.read_csv(data_path)

    # ✅ Force city to stay categorical
    if "city" in df.columns:
        df["city"] = df["city"].astype(str)

    # Drop unwanted columns
    drop_cols = ["latitude", "longitude", "listed_day"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    



    # -------------------------------
    # -------------------------------

    # Define column groups
    # 1) Categorical to be embedded (keep this small & explicit)
    embed_cols = [c for c in ["make_name", "model_name"] if c in df.columns]

    # Clean up raw strings before encoding
    for col in embed_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()

    # 2) Convert bools to 0/1 so they count as numeric later
    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()
    if bool_cols:
        df[bool_cols] = df[bool_cols].astype("int8")

    # 3) Pre-expanded dummy columns already in the CSV (0/1), e.g. body_type_*, engine_type_*, theft_title_*
    dummy_prefixes = ("body_type_", "engine_type_", "theft_title_")
    dummy_cols = [c for c in df.columns if c.startswith(dummy_prefixes)]

    # 4) One-hot columns = all remaining object/category columns EXCEPT the embed columns
    onehot_cols = [
        c for c in df.select_dtypes(include=["object", "category"]).columns
        if c not in embed_cols
    ]

    # 5) Numeric columns = every numeric column (ints/floats) except target and embed columns
    num_cols = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) and c not in embed_cols and c != "price"
    ]


    # Ensure dummy cols are included in numeric (they already should be numeric 0/1)
    # and remove any accidental duplicates while preserving order
    num_cols = list(dict.fromkeys(num_cols + dummy_cols))

    # --- optional: visibility to debug once ---
    print("embed_cols:", embed_cols)
    print("onehot_cols (first 15):", onehot_cols[:15], f"... total={len(onehot_cols)}")
    print("num_cols (first 15):", num_cols[:15], f"... total={len(num_cols)}")



   



    # # -------------------------------
    # # Define column groups
    # # -------------------------------
    # embed_cols = ["make_name", "model_name"]
    # onehot_cols = ['city', 'fleet', 'frame_damaged', 'fuel_type', 'has_accidents', 'isCab', 
    #                'salvage', 'transmission', 'trim_name', 'wheel_system', 
    #                'wheel_system_display']
    
    # num_cols = ['daysonmarket', 'franchise_dealer', 'is_new',
    #             'mileage', 'owner_count', 'price', 'savings_amount', 'sp_id', 'year', 
    #             'body_type_Convertible', 'body_type_Coupe', 'body_type_Hatchback', 'body_type_Minivan', 
    #             'body_type_Pickup Truck', 'body_type_SUV / Crossover', 'body_type_Sedan', 
    #             'body_type_Unknown', 'body_type_Van', 'body_type_Wagon', 'engine_type_H4', 'engine_type_H4 Hybrid', 
    #             'engine_type_H6', 'engine_type_I2', 'engine_type_I3', 'engine_type_I4', 'engine_type_I4 Diesel', 
    #             'engine_type_I4 Flex Fuel Vehicle', 'engine_type_I4 Hybrid', 'engine_type_I5', 
    #             'engine_type_I5 Biodiesel', 'engine_type_I6', 'engine_type_I6 Diesel', 'engine_type_R2', 
    #             'engine_type_Unknown', 'engine_type_V10', 'engine_type_V12', 'engine_type_V6', 'engine_type_V6 Biodiesel', 
    #             'engine_type_V6 Diesel', 'engine_type_V6 Flex Fuel Vehicle', 'engine_type_V6 Hybrid', 'engine_type_V8', 
    #             'engine_type_V8 Biodiesel', 'engine_type_V8 Compressed Natural Gas', 'engine_type_V8 Diesel', 
    #             'engine_type_V8 Flex Fuel Vehicle', 'engine_type_V8 Hybrid', 'engine_type_W12', 'engine_type_W12 Flex Fuel Vehicle', 
    #             'theft_title_False', 'theft_title_True', 'theft_title_Unknown'] # , "latitude", "longitude"

    # -------------------------------
    # Encode embedding columns
    # -------------------------------
 


    # -------------------------------
    # Encode embedding columns (make/model)
    # -------------------------------
    # Ensure these are never numeric-coded before encoding
    # Initialize a dict to hold all encoders
    encoders = {}

    for col in embed_cols:
        le = LabelEncoder()
        clean_vals = df[col].astype(str).str.strip().str.lower()
        le.fit(clean_vals)
        encoders[col] = le           # ✅ save encoder into dict
        df[col] = le.transform(clean_vals)



        # Force all onehot columns to lowercase trimmed strings (case-insensitive consistency)
        for col in onehot_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.lower()
    # -------------------------------
    # One-hot encode smaller categoricals
    # -------------------------------
    onehot_enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    onehot_enc.fit(df[onehot_cols])
    onehot_feature_names = onehot_enc.get_feature_names_out(onehot_cols)

    # -------------------------------
    # Scale numeric features
    # -------------------------------
    scaler = StandardScaler()
    scaler.fit(df[num_cols])

    # -------------------------------
    # Save preprocessing artifacts
    # -------------------------------
    # After fitting encoders/scaler
    feature_order = num_cols + list(onehot_enc.get_feature_names_out(onehot_cols))

    preprocessed = {
        "encoders": encoders,          # ✅ actual dict, not class
        "onehot_enc": onehot_enc,
        "onehot_feature_names": onehot_feature_names,
        "scaler": scaler,
        "embed_cols": embed_cols,
        "onehot_cols": onehot_cols,
        "num_cols": num_cols
    }

      
    print("make_name classes:", encoders["make_name"].classes_[:20])
    print("model_name classes:", encoders["model_name"].classes_[:20])
    # print("make_name classes:", label_encoders["make_name"].classes_[:20])
    # print("model_name classes:", label_encoders["model_name"].classes_[:20])
    # print("DEBUG type of label_encoders:", type(label_encoders))


    joblib.dump(preprocessed, out_path)
    print(f"✅ Preprocessing complete. Saved to: {out_path}")

if __name__ == "__main__":
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(PROJECT_ROOT, "data", "used_cars", "used_cars_data_clean.csv")
    OUT_PATH = os.path.join(PROJECT_ROOT, "models", "preprocessed.pkl")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    preprocess_data(DATA_PATH, OUT_PATH)
