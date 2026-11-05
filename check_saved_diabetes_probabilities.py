import joblib
import pandas as pd
import numpy as np

print("=" * 70)
print("SAVED DIABETES MODEL - PROBABILITY DISTRIBUTION")
print("=" * 70)

model = joblib.load("models/diabetes_model.pkl")
scaler = joblib.load("models/diabetes_scaler.pkl")

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

# Keep only required columns
X = data[features].copy()

# Encode categorical values using the same maps
maps = joblib.load("models/encoding_maps.pkl")

X["gender"] = X["gender"].map(
    maps["diabetes_gender"]
)

X["smoking_history"] = X["smoking_history"].map(
    maps["diabetes_smoking_history"]
)

# Remove rows with unknown encoding
X = X.dropna()

# Scale
X_scaled = scaler.transform(X)

# Predict probabilities
probabilities = model.predict_proba(X_scaled)[:, 1]

print("\nNumber of samples:", len(probabilities))

print("\nProbability statistics:")
print(f"Minimum : {probabilities.min() * 100:.2f}%")
print(f"Maximum : {probabilities.max() * 100:.2f}%")
print(f"Mean    : {probabilities.mean() * 100:.2f}%")
print(f"Median  : {np.median(probabilities) * 100:.2f}%")

print("\nProbability distribution:")

ranges = [
    (0, 10),
    (10, 20),
    (20, 30),
    (30, 40),
    (40, 50),
    (50, 60),
    (60, 70),
    (70, 80),
    (80, 90),
    (90, 100.01)
]

for low, high in ranges:
    count = np.sum(
        (probabilities * 100 >= low) &
        (probabilities * 100 < high)
    )

    print(f"{low:>2}-{high if high < 100.01 else 100:.0f}% : {count}")

print("\n" + "=" * 70)