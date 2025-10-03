import os
from src.train import train_model

if __name__ == "__main__":
    # Project root
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

    # Paths
    DATA_PATH = os.path.join(PROJECT_ROOT, "data", "used_cars", "used_cars_data_clean.csv")
    MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
    MODEL_PATH = os.path.join(MODEL_DIR, "rf_model.pkl")

    # Ensure models directory exists
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Train model
    print("🚀 Training model...")
    model = train_model(DATA_PATH, model_out=MODEL_PATH)

    print(f"✅ Training complete. Model saved to: {MODEL_PATH}")


