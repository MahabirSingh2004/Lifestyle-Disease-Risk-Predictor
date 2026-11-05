import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# ============================================================
# PATH
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_FILE = os.path.join(
    BASE_DIR,
    "datasets",
    "heart_disease_dataset.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_FILE)

df = df.drop_duplicates()
df.columns = df.columns.str.strip()


# ============================================================
# CLEAN DATA
# ============================================================

categorical_columns = [
    "Gender",
    "Smoking",
    "Alcohol Intake",
    "Family History",
    "Diabetes",
    "Obesity",
    "Exercise Induced Angina",
    "Chest Pain Type"
]

for column in categorical_columns:

    # Convert everything to string first
    df[column] = (
        df[column]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
    )


# ============================================================
# CREATE ENCODING MAPS
# ============================================================

maps = {}

for column in categorical_columns:

    values = sorted(
        df[column].unique().tolist(),
        key=lambda x: str(x)
    )

    maps[column] = {
        value: index
        for index, value in enumerate(values)
    }

    df[column] = df[column].map(maps[column])


# ============================================================
# FEATURES AND TARGET
# ============================================================

X = df.drop("Heart Disease", axis=1)

y = df["Heart Disease"]


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# SCALING
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


# ============================================================
# LOGISTIC REGRESSION
# ============================================================

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(
    X_train_scaled,
    y_train
)


# ============================================================
# TEST CASES
# ============================================================

heart_low = np.array([[
    30, 0, 170, 110, 65, 2, 1, 5,
    0, 0, 0, 2, 90, 0, 3
]])

heart_medium = np.array([[
    50, 0, 210, 130, 75, 2, 1, 3,
    0, 0, 0, 5, 110, 0, 1
]])

heart_high = np.array([[
    65, 1, 280, 160, 95, 0, 0, 0.5,
    1, 1, 1, 9, 180, 1, 2
]])


# ============================================================
# TEST PROBABILITIES
# ============================================================

print("=" * 70)
print("HEART DISEASE - LOGISTIC REGRESSION TEST")
print("=" * 70)

for name, data in [
    ("LOW", heart_low),
    ("MODERATE", heart_medium),
    ("HIGH", heart_high)
]:

    scaled = scaler.transform(data)

    prediction = model.predict(scaled)[0]

    probability = model.predict_proba(scaled)[0][1]

    print(
        f"{name:10} -> "
        f"Prediction: {prediction} | "
        f"Risk: {probability * 100:.2f}%"
    )


print("\n" + "=" * 70)