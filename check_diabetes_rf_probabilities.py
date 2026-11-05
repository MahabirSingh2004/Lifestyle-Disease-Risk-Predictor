import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

print("=" * 70)
print("DIABETES RANDOM FOREST - REAL DATA PROBABILITY CHECK")
print("=" * 70)

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

df = pd.read_csv("datasets/diabetes_pred_dataset.csv")

df.columns = df.columns.str.strip()
df = df.drop_duplicates()

# ------------------------------------------------------------
# ENCODE CATEGORICAL FEATURES
# ------------------------------------------------------------

df["gender"] = df["gender"].astype(str).str.strip()
df["smoking_history"] = df["smoking_history"].astype(str).str.strip()

gender_map = {
    value: index
    for index, value in enumerate(sorted(df["gender"].unique()))
}

smoking_map = {
    value: index
    for index, value in enumerate(sorted(df["smoking_history"].unique()))
}

df["gender"] = df["gender"].map(gender_map)
df["smoking_history"] = df["smoking_history"].map(smoking_map)

# ------------------------------------------------------------
# FEATURES
# ------------------------------------------------------------

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

X = df[features]
y = df["diabetes"]

# ------------------------------------------------------------
# TRAIN / TEST SPLIT
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ------------------------------------------------------------
# SCALING
# ------------------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------------------------------------
# RANDOM FOREST
# ------------------------------------------------------------

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_scaled, y_train)

# ------------------------------------------------------------
# TEST-SET PROBABILITIES
# ------------------------------------------------------------

probabilities = model.predict_proba(X_test_scaled)[:, 1]

print("\nProbability statistics:")
print("-" * 70)

print(f"Minimum probability : {probabilities.min() * 100:.2f}%")
print(f"Maximum probability : {probabilities.max() * 100:.2f}%")
print(f"Mean probability    : {probabilities.mean() * 100:.2f}%")
print(f"Median probability  : {np.median(probabilities) * 100:.2f}%")

# ------------------------------------------------------------
# PROBABILITY RANGES
# ------------------------------------------------------------

print("\nProbability ranges:")
print("-" * 70)

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
    (90, 100)
]

for low, high in ranges:

    count = np.sum(
        (probabilities * 100 >= low) &
        (probabilities * 100 < high)
    )

    print(f"{low:>2}-{high:<3}% : {count}")

print("\n" + "=" * 70)