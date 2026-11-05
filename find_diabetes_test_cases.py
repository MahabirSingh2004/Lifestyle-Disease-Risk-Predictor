import joblib
import pandas as pd
import numpy as np

print("=" * 70)
print("FINDING REALISTIC DIABETES TEST CASES")
print("=" * 70)

model = joblib.load("models/diabetes_model.pkl")
scaler = joblib.load("models/diabetes_scaler.pkl")
maps = joblib.load("models/encoding_maps.pkl")

data = pd.read_csv("datasets/diabetes_pred_dataset.csv")

features = [
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "smoking_history",
    "bmi",
    "HbA1c_level",
    "blood_glucose_level"
]

X = data[features].copy()

X["gender"] = X["gender"].map(
    maps["diabetes_gender"]
)

X["smoking_history"] = X["smoking_history"].map(
    maps["diabetes_smoking_history"]
)

valid = X.notna().all(axis=1)

X = X[valid]
original = data.loc[X.index].copy()

X_scaled = scaler.transform(X)

probabilities = model.predict_proba(X_scaled)[:, 1]

original["Predicted_Risk"] = probabilities * 100

# Find cases closest to selected risk levels
targets = [5, 25, 50, 75, 95]

print("\n" + "=" * 70)
print("REALISTIC TEST CASES")
print("=" * 70)

for target in targets:

    index = (original["Predicted_Risk"] - target).abs().idxmin()

    row = original.loc[index]

    print(f"\nTARGET RISK: {target}%")
    print(f"Actual model risk: {row['Predicted_Risk']:.2f}%")

    for feature in features:
        print(f"{feature:<22}: {row[feature]}")

print("\n" + "=" * 70)