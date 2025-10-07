# train_embedding.py
import os
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd

# -------------------------------
# Dataset Class
# -------------------------------
class CarDataset(Dataset):
    def __init__(self, embed_data, other_data, y):
        self.embed_data = torch.tensor(embed_data, dtype=torch.long)
        self.other_data = torch.tensor(other_data, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self): 
        return len(self.y)

    def __getitem__(self, idx): 
        return self.embed_data[idx], self.other_data[idx], self.y[idx]


# -------------------------------
# Model
# -------------------------------
class CarModel(nn.Module):
    def __init__(self, embed_sizes, n_other):
        super().__init__()
        self.embeds = nn.ModuleList([nn.Embedding(cat, dim) for cat, dim in embed_sizes])
        self.fc1 = nn.Linear(sum(dim for _, dim in embed_sizes) + n_other, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x_embed, x_other):
        embs = [emb(x_embed[:, i]) for i, emb in enumerate(self.embeds)]
        x = torch.cat(embs + [x_other], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# -------------------------------
# Training Script
# -------------------------------
if __name__ == "__main__":
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(MODEL_DIR, exist_ok=True)

    PREP_PATH = os.path.join(MODEL_DIR, "preprocessed.pkl")
    preprocessed = joblib.load(PREP_PATH)

    # Column groups
    embed_cols = preprocessed["embed_cols"]
    onehot_cols = preprocessed["onehot_cols"]
    num_cols = preprocessed["num_cols"]

    # Encoders and transformers
    encoders = preprocessed["encoders"]
    onehot_enc = preprocessed["onehot_enc"]
    scaler = preprocessed["scaler"]

    # -------------------------------
    # Load raw dataset
    # -------------------------------
    DATA_PATH = os.path.join(PROJECT_ROOT, "data", "used_cars", "used_cars_data_clean.csv")
    df = pd.read_csv(DATA_PATH)

    # -------------------------------
    # Clean and normalize columns before transform
    # -------------------------------
    for c in embed_cols + onehot_cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().str.lower()

    # Target variable
    y = np.log1p(df["price"].values)

    # --- Ensure numeric types ---
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Convert dummy columns (body_type_*, engine_type_*, theft_title_*)
    df = df.astype({
        c: float for c in df.columns 
        if c.startswith(("body_type_", "engine_type_", "theft_title_"))
    })

    # -------------------------------
    # Encode embedding columns safely
    # -------------------------------
    for col in embed_cols:
        known_classes = set(encoders[col].classes_)
        unseen = set(df[col].unique()) - known_classes
        if unseen:
            print(f"⚠️ Dropping {len(unseen)} unseen labels in '{col}': {list(unseen)[:10]}...")
            df = df[df[col].isin(known_classes)]

    # Transform using saved encoders
    X_embed = np.column_stack([
        encoders[col].transform(df[col]) for col in embed_cols
    ]).astype(np.int64)

    # -------------------------------
    # One-hot encode smaller categoricals
    # -------------------------------
    for col in onehot_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
    X_onehot = onehot_enc.transform(df[onehot_cols]).astype(np.float32)

    # -------------------------------
    # Scale numeric features (no re-fitting!)
    # -------------------------------
    X_num = scaler.transform(df[num_cols]).astype(np.float32)

    # -------------------------------
    # Combine numeric + one-hot
    # -------------------------------
    X_other = np.hstack([X_num, X_onehot]).astype(np.float32)

    # Debug info
    print("DEBUG (TRAIN):")
    print("X_embed shape:", X_embed.shape, "| dtype:", X_embed.dtype)
    print("X_num shape:", X_num.shape)
    print("X_onehot shape:", X_onehot.shape)
    print("X_other shape:", X_other.shape)

    # -------------------------------
    # Train/Validation Split
    # -------------------------------
    X_embed_train, X_embed_val, X_other_train, X_other_val, y_train, y_val = train_test_split(
        X_embed, X_other, y, test_size=0.2, random_state=42
    )

    # Datasets & Loaders
    train_ds = CarDataset(X_embed_train, X_other_train, y_train)
    val_ds = CarDataset(X_embed_val, X_other_val, y_val)
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=64)

    # Embedding sizes
    embed_sizes = [
        (len(encoders["make_name"].classes_), 50),
        (len(encoders["model_name"].classes_), 100)
    ]

    # Model setup
    model = CarModel(embed_sizes, X_other.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # -------------------------------
    # Training Loop
    # -------------------------------
    EPOCHS = 30
    for epoch in range(EPOCHS):
        model.train()
        for xb_embed, xb_other, yb in train_dl:
            optimizer.zero_grad()
            preds = model(xb_embed, xb_other)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_losses, preds_all, y_all = [], [], []
        with torch.no_grad():
            for xb_embed, xb_other, yb in val_dl:
                preds = model(xb_embed, xb_other)
                val_losses.append(criterion(preds, yb).item())
                preds_all.append(preds.numpy())
                y_all.append(yb.numpy())
        val_loss = np.mean(val_losses)

        preds_all = np.vstack(preds_all).flatten()
        y_all = np.vstack(y_all).flatten()
        r2 = r2_score(y_all, preds_all)
        median_pred = np.median(np.expm1(preds_all))

        print(f"Epoch {epoch+1}: Val MSE={val_loss:.2f}, R²={r2:.4f}, Median Pred=${median_pred:,.2f}")

    # -------------------------------
    # Save Model
    # -------------------------------
    MODEL_PATH = os.path.join(MODEL_DIR, "deep_model.pth")
    torch.save(model.state_dict(), MODEL_PATH)

    preprocessed["input_size"] = X_other.shape[1]
    joblib.dump(preprocessed, PREP_PATH)

    print(f"✅ Model saved to {MODEL_PATH}")
    print(f"✅ Preprocessing updated with input_size={X_other.shape[1]}")
