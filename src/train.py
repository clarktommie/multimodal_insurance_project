import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor

def train_model(data_path, model_out="rf_model.pkl"):
    df = pd.read_csv(data_path)

    # Target
    y = df["price"]
    X = df.drop(columns=["price"])

    # -------------------------------
    # Define feature groups
    # -------------------------------
    # Remove body_type (since it's already expanded in the CSV)
    categorical_features = [
    "make_name", "model_name", "engine_type", "engine_cylinders",
    "wheel_system_display", "fuel_type", "transmission_display",
    "trim_name", "exterior_color", "city",
    "theft_title", "salvage", "frame_damaged", "has_accidents",
    "isCab", "fleet"
    ]

    # Numeric features (keep your list, but note: body_type_* dummies will automatically be numeric!)
    numeric_features = [
    "mileage", "owner_count", "length", "width", "wheelbase", "maximum_seating",
    "daysonmarket", "year", "listed_year", "listed_month",
    "latitude", "longitude", "body_type_Convertible", "body_type_Coupe", "body_type_SUV / Crossover",
    "body_type_Sedan", "body_type_Unknown"
    ]



    # # ✅ Keep only features that actually exist in this dataset
    # categorical_features = [c for c in categorical_features if c in X.columns]
    # numeric_features = [c for c in numeric_features if c in X.columns]

    # print("Final categorical features:", categorical_features)
    # print("Final numeric features:", numeric_features)

    # -------------------------------
    # Preprocessing
    # -------------------------------
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", categorical_transformer, categorical_features),
            ("numeric", numeric_transformer, numeric_features)
        ]
    )

    # -------------------------------
    # Model pipeline
    # -------------------------------
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    # -------------------------------
    # Split
    # -------------------------------
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -------------------------------
    # Train
    # -------------------------------
    print("PLEASE WAIT.... Training model...")
    model.fit(X_train, y_train)

    # -------------------------------
    # Save
    # -------------------------------
    joblib.dump(model, model_out)
    print(f"DONE!! Model saved to {model_out}")

    return model
