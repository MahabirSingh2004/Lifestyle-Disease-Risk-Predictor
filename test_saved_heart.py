import os
import pickle
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)


# ============================================================
# LOAD MODEL
# ============================================================

model_path = os.path.join(
    MODEL_DIR,
    "heart_model.pkl"
)

scaler_path = os.path.join(
    MODEL_DIR,
    "heart_scaler.pkl"
)

maps_path = os.path.join(
    MODEL_DIR,
    "encoding_maps.pkl"
)


with open(model_path, "rb") as file:
    model = pickle.load(file)

with open(scaler_path, "rb") as file:
    scaler = pickle.load(file)

with open(maps_path, "rb") as file:
    maps = pickle.load(file)


# ============================================================
# DISPLAY MODEL
# ============================================================

print("=" * 70)
print("SAVED HEART DISEASE MODEL TEST")
print("=" * 70)

print(
    "\nModel:",
    type(model).__name__
)

print(
    "Scaler:",
    type(scaler).__name__
)


# ============================================================
# GET HEART ENCODING MAPS
# ============================================================

print("\nHeart encoding maps:")

for key, value in maps.items():

    if key.startswith("heart_"):

        print(
            f"{key}: {value}"
        )


# ============================================================
# FEATURE NAMES
# ============================================================

features = [
    "Age",
    "Gender",
    "Cholesterol",
    "Blood Pressure",
    "Heart Rate",
    "Smoking",
    "Alcohol Intake",
    "Exercise Hours",
    "Family History",
    "Diabetes",
    "Obesity",
    "Stress Level",
    "Blood Sugar",
    "Exercise Induced Angina",
    "Chest Pain Type"
]


# ============================================================
# TEST CASES
# ============================================================

heart_low = pd.DataFrame([[
    30,
    0,
    170,
    110,
    65,
    2,
    1,
    5,
    0,
    0,
    0,
    2,
    90,
    0,
    3
]], columns=features)


heart_medium = pd.DataFrame([[
    50,
    0,
    210,
    130,
    75,
    2,
    1,
    3,
    0,
    0,
    0,
    5,
    110,
    0,
    1
]], columns=features)


heart_high = pd.DataFrame([[
    65,
    1,
    280,
    160,
    95,
    0,
    0,
    0.5,
    1,
    1,
    1,
    9,
    180,
    1,
    2
]], columns=features)


# ============================================================
# TEST FUNCTION
# ============================================================

def test_case(name, data):

    scaled = scaler.transform(data)

    prediction = model.predict(
        scaled
    )[0]

    probabilities = model.predict_proba(
        scaled
    )[0]

    risk = probabilities[1] * 100

    print(
        f"{name:10} -> "
        f"Prediction: {prediction} | "
        f"Risk: {risk:.2f}%"
    )


# ============================================================
# RUN TESTS
# ============================================================

print("\n" + "=" * 70)
print("RISK RESULTS")
print("=" * 70)

test_case(
    "LOW",
    heart_low
)

test_case(
    "MODERATE",
    heart_medium
)

test_case(
    "HIGH",
    heart_high
)

print("\n" + "=" * 70)