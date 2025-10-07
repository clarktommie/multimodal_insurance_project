import os
from src.train import train_model

if __name__ == "__main__":
    # Project root
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

    # Paths
    DATA_PATH = os.path.join(PROJECT_ROOT, "data", "used_cars", "used_cars_data_clean.csv")
    MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
    PREPROC_PATH = os.path.join(MODEL_DIR, "preprocessed.pkl")

    # Ensure models directory exists
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Run preprocessing + save encoders/data splits
    print("🚀 Preparing data + encoders for embedding model...")
    data_dict = train_model(DATA_PATH, model_out=PREPROC_PATH)

    print(f"✅ Preprocessing complete. Saved to: {PREPROC_PATH}")
