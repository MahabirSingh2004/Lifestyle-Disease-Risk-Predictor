import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_FILE = os.path.join(
    BASE_DIR,
    "datasets",
    "Health_risk_Predictor.csv"
)

df = pd.read_csv(DATA_FILE)

print("=" * 80)
print("HEALTH RISK DATASET COLUMNS")
print("=" * 80)

print(f"\nShape: {df.shape}\n")

for i, column in enumerate(df.columns, start=1):
    print(f"{i:2}. {column}")

print("\n" + "=" * 80)
print("FIRST 3 ROWS")
print("=" * 80)

print(df.head(3).to_string())

print("\n" + "=" * 80)