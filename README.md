# Multimodal Insurance Project 🚗📊🖼️

This project explores **multimodal machine learning** for assessing **used car insurance risk and pricing**.  
It begins with a **tabular baseline model** and will later incorporate **text** (descriptions, accident reports) and **images** (car photos).

---

## 🚀 Current Status (Baseline Model)

- Implemented baseline **RandomForestRegressor** on **tabular data** only.
- Columns used after cleaning and feature selection:

**Categorical features:**

```
['make_name', 'model_name', 'engine_type', 'engine_cylinders']
```

**Numeric features:**

```
['mileage', 'owner_count']
```

- Training run command:

```bash
uv run main.py
```

- Example output:

```
🚀 Training model...
Final categorical features: ['make_name', 'model_name', 'engine_type', 'engine_cylinders']
Final numeric features: ['mileage', 'owner_count']
✅ Model saved to models/rf_model.pkl
```

- Model is saved to:

```
models/rf_model.pkl
```

---

## 📂 Project Structure

```
multimodal-insurance-project/
│── data/                # Raw and cleaned datasets (not pushed to GitHub)
│── models/              # Saved models (.pkl)
│── notebooks/           # Jupyter notebooks for exploration
│── src/                 # Source code (train, preprocess, utils)
│── streamlit/           # Streamlit app for deployment
│── modal/               # Modal deployment scripts
│── supabase/            # Supabase client for experiment tracking
│── main.py              # Entry point for training
│── README.md            # Project documentation
```

---

## 🔜 Next Steps

- Add evaluation metrics (**MSE, R²**) and feature importances.
- Extend pipeline to include **text features** (NLP embeddings from descriptions, accident text).
- Add **image modality** (Stanford Cars dataset + CNN embeddings).
- Integrate with **Streamlit** for an interactive UI.
- Deploy using **Modal** and store experiment results in **Supabase**.

---

## 💡 ROI (Why This Matters)

- Insurance pricing and risk assessment can be improved with multimodal inputs.
- Tabular alone gives a baseline, but text + image data will likely reduce error and improve fairness.
- Demonstrates a **full-stack ML workflow**: data → model → deployment → monitoring.

---
