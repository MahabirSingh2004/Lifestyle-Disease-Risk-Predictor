import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

print("=" * 70)
print("DIABETES - RANDOM FOREST PROBABILITY TEST")
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
# TEST CASES
# ------------------------------------------------------------

low = pd.DataFrame([[
    0,
    25,
    0,
    0,
    4,
    21.5,
    4.8,
    90
]], columns=features)

moderate = pd.DataFrame([[
    0,
    45,
    0,
    0,
    3,
    27.5,
    5.8,
    120
]], columns=features)

high = pd.DataFrame([[
    1,
    65,
    1,
    1,
    1,
    34.5,
    8.5,
    200
]], columns=features)


# ------------------------------------------------------------
# TEST FUNCTION
# ------------------------------------------------------------

def test_case(name, data):

    scaled = scaler.transform(data)

    prediction = model.predict(scaled)[0]
    probability = model.predict_proba(scaled)[0][1]

    print(
        f"{name:<10} -> "
        f"Prediction: {prediction} | "
        f"Risk: {probability * 100:.2f}%"
    )


# ------------------------------------------------------------
# RESULTS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("RISK RESULTS")
print("=" * 70)

test_case("LOW", low)
test_case("MODERATE", moderate)
test_case("HIGH", high)

print("\n" + "=" * 70)