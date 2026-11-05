import joblib
import pandas as pd

print("=" * 70)
print("SAVED DIABETES MODEL TEST")
print("=" * 70)

# ------------------------------------------------------------
# LOAD MODEL, SCALER AND ENCODING MAPS
# ------------------------------------------------------------

model = joblib.load("models/diabetes_model.pkl")
scaler = joblib.load("models/diabetes_scaler.pkl")
maps = joblib.load("models/encoding_maps.pkl")

print("\nModel :", type(model).__name__)
print("Scaler:", type(scaler).__name__)

print("\nDiabetes encoding maps:")

for key, value in maps.items():
    if key.startswith("diabetes_"):
        print(f"{key}: {value}")


# ------------------------------------------------------------
# EXACT FEATURES USED DURING TRAINING
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


# ------------------------------------------------------------
# TEST CASES
# ------------------------------------------------------------

diabetes_low = pd.DataFrame([[
    0,      # gender
    25,     # age
    0,      # hypertension
    0,      # heart disease
    4,      # smoking history
    21.5,   # BMI
    4.8,    # HbA1c
    90      # blood glucose
]], columns=features)


diabetes_moderate = pd.DataFrame([[
    0,      # gender
    45,     # age
    0,      # hypertension
    0,      # heart disease
    3,      # smoking history
    27.5,   # BMI
    5.8,    # HbA1c
    120     # blood glucose
]], columns=features)


diabetes_high = pd.DataFrame([[
    1,      # gender
    65,     # age
    1,      # hypertension
    1,      # heart disease
    1,      # smoking history
    34.5,   # BMI
    8.5,    # HbA1c
    200     # blood glucose
]], columns=features)


# ------------------------------------------------------------
# FUNCTION TO TEST MODEL
# ------------------------------------------------------------

def test_case(name, data):

    scaled_data = scaler.transform(data)

    prediction = model.predict(scaled_data)[0]
    probability = model.predict_proba(scaled_data)[0][1]

    print(
        f"{name:<10} -> "
        f"Prediction: {prediction} | "
        f"Risk: {probability * 100:.2f}%"
    )


# ------------------------------------------------------------
# RUN TESTS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("RISK RESULTS")
print("=" * 70)

test_case("LOW", diabetes_low)
test_case("MODERATE", diabetes_moderate)
test_case("HIGH", diabetes_high)

print("\n" + "=" * 70)