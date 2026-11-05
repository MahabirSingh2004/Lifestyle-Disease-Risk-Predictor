import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "datasets")


# ============================================================
# HEART DATASET
# ============================================================

heart_file = os.path.join(
    DATA_DIR,
    "heart_disease_dataset.csv"
)

heart = pd.read_csv(heart_file)

print("=" * 70)
print("HEART DISEASE DATASET")
print("=" * 70)

print("\nShape:")
print(heart.shape)

print("\nTarget distribution:")
print(
    heart["Heart Disease"]
    .value_counts()
)

print("\nTarget percentage:")
print(
    heart["Heart Disease"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\nColumns:")
print(list(heart.columns))


# ============================================================
# DIABETES DATASET
# ============================================================

diabetes_file = os.path.join(
    DATA_DIR,
    "diabetes_pred_dataset.csv"
)

diabetes = pd.read_csv(diabetes_file)

print("\n\n" + "=" * 70)
print("DIABETES DATASET")
print("=" * 70)

print("\nShape:")
print(diabetes.shape)

print("\nTarget distribution:")
print(
    diabetes["diabetes"]
    .value_counts()
)

print("\nTarget percentage:")
print(
    diabetes["diabetes"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\nColumns:")
print(list(diabetes.columns))