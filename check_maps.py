import joblib
import os
import pprint

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

encoding_maps = joblib.load(
    os.path.join(
        MODEL_DIR,
        "encoding_maps.pkl"
    )
)

print("\n" + "=" * 70)
print("ACTUAL ENCODING MAPS")
print("=" * 70)

for name, mapping in encoding_maps.items():

    print("\n" + "-" * 70)
    print(name)
    print("-" * 70)

    pprint.pprint(mapping)

print("\n" + "=" * 70)