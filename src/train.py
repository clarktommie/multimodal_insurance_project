import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
import joblib


def train_model(data_path, model_out="rf_model.pkl"):
    # Load dataset
    df = pd.read_csv(data_path)

    # Target
    y = df["price"]
    X = df.drop(columns=["price"])

    # Define candidate feature lists
    categorical_features = [
        "make_name", "model_name", "engine_type", "engine_cylinders",
        "wheel_system_display", "has_accidents", "salvage", "frame_damaged",
        "isCab", "fleet", "theft_title"
    ]

    numeric_features = [
        "mileage", "owner_count", "length", "width", "wheelbase", "maximum_seating"
    ]

    # ✅ Keep only features that actually exist in dataset
    categorical_features = [c for c in categorical_features if c in X.columns]
    numeric_features = [c for c in numeric_features if c in X.columns]

    print("Final categorical features:", categorical_features)
    print("Final numeric features:", numeric_features)

    # Preprocessing
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

    # Model pipeline
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train
    print("🚀 Training model...")
    model.fit(X_train, y_train)

    # Save
    joblib.dump(model, model_out)
    print(f"✅ Model saved to {model_out}")

    return model
