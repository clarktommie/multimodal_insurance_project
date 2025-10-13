import pandas as pd

# Load the feature list
df = pd.read_csv("all_features_expected.csv")

# Create a blank row (0 for numeric/onehot)
sample = pd.Series(0.0, index=df["feature_name"].tolist())

# Add embedding columns (these are handled separately)
sample["make_name"] = "land rover"
sample["model_name"] = "discovery sport"

# Set some realistic numeric values
sample.update({
    "year": 2025,
    "mileage": 10000,
    "owner_count": 0,
    "daysonmarket": 18,
    "is_new": 1,
    "franchise_dealer": 1,
    "listed_year": 2025,
    "listed_month": 7,
    "body_type_SUV / Crossover": 1,
    "engine_type_I4": 1,
    "theft_title_False": 1,
})

# Save
sample.to_csv("sample_car_input.csv", header=True)
print("✅ Created sample_car_input.csv — ready for prediction")
