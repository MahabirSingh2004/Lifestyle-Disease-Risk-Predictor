import os
import sys
import joblib
import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")


# ============================================================
# LOAD MODELS
# ============================================================

diabetes_model = joblib.load(
    os.path.join(MODEL_DIR, "diabetes_model.pkl")
)

heart_model = joblib.load(
    os.path.join(MODEL_DIR, "heart_model.pkl")
)

sleep_model = joblib.load(
    os.path.join(MODEL_DIR, "sleep_model.pkl")
)


# ============================================================
# LOAD SCALERS
# ============================================================

diabetes_scaler = joblib.load(
    os.path.join(MODEL_DIR, "diabetes_scaler.pkl")
)

heart_scaler = joblib.load(
    os.path.join(MODEL_DIR, "heart_scaler.pkl")
)

sleep_scaler = joblib.load(
    os.path.join(MODEL_DIR, "sleep_scaler.pkl")
)


# ============================================================
# LOAD ENCODING MAPS
# ============================================================

encoding_maps = joblib.load(
    os.path.join(MODEL_DIR, "encoding_maps.pkl")
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("NEW MODEL PROBABILITY TEST")
print("=" * 70)


# ============================================================
# DIABETES
# ============================================================

print("\n" + "=" * 70)
print("DIABETES")
print("=" * 70)


diabetes_features = [
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "smoking_history",
    "bmi",
    "HbA1c_level",
    "blood_glucose_level"
]


# Low-risk test case
diabetes_low = pd.DataFrame([[
    0,
    25,
    0,
    0,
    4,
    21.5,
    4.8,
    90
]], columns=diabetes_features)


# Moderate test case
diabetes_moderate = pd.DataFrame([[
    0,
    45,
    0,
    0,
    3,
    27.5,
    5.8,
    120
]], columns=diabetes_features)


# High-risk test case
diabetes_high = pd.DataFrame([[
    1,
    65,
    1,
    1,
    1,
    34.5,
    8.5,
    200
]], columns=diabetes_features)


for name, data in [
    ("LOW", diabetes_low),
    ("MODERATE", diabetes_moderate),
    ("HIGH", diabetes_high)
]:

    scaled = diabetes_scaler.transform(data)

    prediction = diabetes_model.predict(scaled)[0]

    probability = diabetes_model.predict_proba(
        scaled
    )[0][1]

    print(
        f"{name:<10} -> "
        f"Prediction: {prediction} | "
        f"Risk: {probability * 100:.2f}%"
    )


# ============================================================
# HEART DISEASE
# ============================================================

print("\n" + "=" * 70)
print("HEART DISEASE")
print("=" * 70)


heart_features = [
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


# Low-risk test case
heart_low = pd.DataFrame([[
    30,
    0,
    180,
    110,
    65,
    2,
    1,
    6,
    0,
    0,
    0,
    2,
    90,
    0,
    0
]], columns=heart_features)


# Moderate-risk test case
heart_moderate = pd.DataFrame([[
    50,
    0,
    230,
    130,
    75,
    1,
    1,
    3,
    1,
    0,
    1,
    5,
    120,
    0,
    1
]], columns=heart_features)


# High-risk test case
heart_high = pd.DataFrame([[
    70,
    1,
    300,
    160,
    95,
    0,
    0,
    1,
    1,
    1,
    1,
    9,
    180,
    1,
    3
]], columns=heart_features)


for name, data in [
    ("LOW", heart_low),
    ("MODERATE", heart_moderate),
    ("HIGH", heart_high)
]:

    scaled = heart_scaler.transform(data)

    prediction = heart_model.predict(scaled)[0]

    probability = heart_model.predict_proba(
        scaled
    )[0][1]

    print(
        f"{name:<10} -> "
        f"Prediction: {prediction} | "
        f"Risk: {probability * 100:.2f}%"
    )


# ============================================================
# SLEEP DISORDER
# ============================================================

print("\n" + "=" * 70)
print("SLEEP DISORDER")
print("=" * 70)


sleep_features = [
    "Gender",
    "Age",
    "Occupation",
    "Sleep Duration",
    "Quality of Sleep",
    "Physical Activity Level",
    "Stress Level",
    "BMI Category",
    "Blood Pressure",
    "Heart Rate",
    "Daily Steps"
]


# Healthy sleep
sleep_healthy = pd.DataFrame([[
    0,
    30,
    9,
    8,
    8,
    80,
    2,
    0,
    6,
    65,
    9000
]], columns=sleep_features)


# Moderate sleep
sleep_medium = pd.DataFrame([[
    0,
    40,
    10,
    6.5,
    6,
    50,
    5,
    3,
    15,
    75,
    6000
]], columns=sleep_features)


# Poor sleep
sleep_poor = pd.DataFrame([[
    1,
    50,
    4,
    4.5,
    3,
    20,
    9,
    2,
    23,
    90,
    2500
]], columns=sleep_features)


for name, data in [
    ("HEALTHY", sleep_healthy),
    ("MODERATE", sleep_medium),
    ("POOR", sleep_poor)
]:

    scaled = sleep_scaler.transform(data)

    prediction = sleep_model.predict(scaled)[0]

    probabilities = sleep_model.predict_proba(
        scaled
    )[0]

    classes = sleep_model.classes_

    predicted_index = list(classes).index(prediction)

    probability = probabilities[predicted_index]

    print(
        f"{name:<10} -> "
        f"Prediction: {prediction} | "
        f"Confidence: {probability * 100:.2f}%"
    )


# ============================================================
# END
# ============================================================

print("\n" + "=" * 70)
print("MODEL TEST COMPLETED")
print("=" * 70)