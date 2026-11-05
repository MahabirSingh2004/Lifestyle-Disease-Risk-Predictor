import os
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
# CLEAN CATEGORICAL COLUMNS
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
    df[column] = (
        df[column]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
    )


# ============================================================
# ENCODE
# ============================================================

for column in categorical_columns:

    values = sorted(
        df[column].unique().tolist(),
        key=lambda x: str(x)
    )

    mapping = {
        value: index
        for index, value in enumerate(values)
    }

    df[column] = df[column].map(mapping)


# ============================================================
# FEATURES / TARGET
# ============================================================

X = df.drop("Heart Disease", axis=1)

y = df["Heart Disease"]


# ============================================================
# SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# SCALE
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
# REAL TEST DATA PROBABILITIES
# ============================================================

probabilities = model.predict_proba(X_test_scaled)[:, 1]


# ============================================================
# SUMMARY
# ============================================================

print("=" * 70)
print("HEART DISEASE - REAL DATA PROBABILITY ANALYSIS")
print("=" * 70)

print(f"\nMinimum probability : {probabilities.min() * 100:.2f}%")
print(f"Maximum probability : {probabilities.max() * 100:.2f}%")
print(f"Mean probability    : {probabilities.mean() * 100:.2f}%")
print(f"Median probability  : {pd.Series(probabilities).median() * 100:.2f}%")


print("\nProbability ranges:")

ranges = [
    ("0-10%", 0.00, 0.10),
    ("10-20%", 0.10, 0.20),
    ("20-30%", 0.20, 0.30),
    ("30-40%", 0.30, 0.40),
    ("40-50%", 0.40, 0.50),
    ("50-60%", 0.50, 0.60),
    ("60-70%", 0.60, 0.70),
    ("70-80%", 0.70, 0.80),
    ("80-90%", 0.80, 0.90),
    ("90-100%", 0.90, 1.00)
]

for label, lower, upper in ranges:

    count = (
        (probabilities >= lower) &
        (probabilities < upper)
    ).sum()

    print(f"{label:10} : {count}")


print("\n" + "=" * 70)