import joblib, os, pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PREP_PATH = os.path.join(PROJECT_ROOT, "models", "preprocessed.pkl")

# Load preprocessing info
preprocessed = joblib.load(PREP_PATH)

num_cols = preprocessed["num_cols"]
onehot_features = preprocessed["onehot_feature_names"]
embed_cols = preprocessed["embed_cols"]

print("\n=== EMBEDDING FEATURES ===")
print(embed_cols)

print("\n=== NUMERIC FEATURES ===")
print(num_cols)

print("\n=== ONE-HOT FEATURES ===")
print(onehot_features[:50], "...")  # print first 50 for preview
print(f"\nTotal one-hot features: {len(onehot_features)}")

# Optional — save full list to CSV so you can fill it in later
pd.DataFrame({
    "feature_name": num_cols + onehot_features.tolist(),

    "type": ["numeric"] * len(num_cols) + ["onehot"] * len(onehot_features)
}).to_csv("all_features_expected.csv", index=False)
print("\n✅ Saved all feature names to all_features_expected.csv")
