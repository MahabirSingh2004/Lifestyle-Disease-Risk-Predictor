import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "datasets")

file_path = os.path.join(
    DATA_DIR,
    "heart_disease_dataset.csv"
)

df = pd.read_csv(file_path)

print("=" * 70)
print("HEART DISEASE FEATURE ANALYSIS")
print("=" * 70)

target = "Heart Disease"


# ============================================================
# NUMERIC FEATURES
# ============================================================

print("\nNUMERIC FEATURES")
print("-" * 70)

numeric_columns = df.select_dtypes(
    include="number"
).columns

correlations = (
    df[numeric_columns]
    .corr()[target]
    .drop(target)
    .sort_values(
        key=abs,
        ascending=False
    )
)

print(
    correlations.round(4)
)


# ============================================================
# CATEGORICAL FEATURES
# ============================================================

print("\n\nCATEGORICAL FEATURES")
print("-" * 70)

categorical_columns = df.select_dtypes(
    include="object"
).columns

for column in categorical_columns:

    print(f"\n--- {column} ---")

    table = pd.crosstab(
        df[column],
        df[target],
        normalize="index"
    ) * 100

    print(
        table.round(2)
    )


# ============================================================
# TARGET GROUP MEANS
# ============================================================

print("\n\nNUMERIC FEATURE MEANS BY TARGET")
print("-" * 70)

means = df.groupby(target)[numeric_columns].mean()

print(
    means.round(2)
)


print("\n" + "=" * 70)
print("ANALYSIS COMPLETED")
print("=" * 70)