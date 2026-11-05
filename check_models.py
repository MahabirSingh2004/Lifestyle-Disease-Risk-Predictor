import joblib
import os

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)


# ============================================================
# LOAD MODELS
# ============================================================

diabetes_model = joblib.load(
    os.path.join(
        MODEL_DIR,
        "diabetes_model.pkl"
    )
)

heart_model = joblib.load(
    os.path.join(
        MODEL_DIR,
        "heart_model.pkl"
    )
)

sleep_model = joblib.load(
    os.path.join(
        MODEL_DIR,
        "sleep_model.pkl"
    )
)


# ============================================================
# DISPLAY MODEL INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("MODEL INFORMATION")
print("=" * 70)


print("\nDIABETES MODEL")
print("-" * 70)
print("Model type:", type(diabetes_model))
print("Classes:", diabetes_model.classes_)

if hasattr(diabetes_model, "n_estimators"):
    print("Number of estimators:", diabetes_model.n_estimators)

if hasattr(diabetes_model, "max_depth"):
    print("Max depth:", diabetes_model.max_depth)


print("\nHEART DISEASE MODEL")
print("-" * 70)
print("Model type:", type(heart_model))
print("Classes:", heart_model.classes_)

if hasattr(heart_model, "n_estimators"):
    print("Number of estimators:", heart_model.n_estimators)

if hasattr(heart_model, "max_depth"):
    print("Max depth:", heart_model.max_depth)

if hasattr(heart_model, "get_depth"):
    print("Tree depth:", heart_model.get_depth())

if hasattr(heart_model, "get_n_leaves"):
    print("Number of leaves:", heart_model.get_n_leaves())


print("\nSLEEP DISORDER MODEL")
print("-" * 70)
print("Model type:", type(sleep_model))
print("Classes:", sleep_model.classes_)

if hasattr(sleep_model, "n_estimators"):
    print("Number of estimators:", sleep_model.n_estimators)

if hasattr(sleep_model, "max_depth"):
    print("Max depth:", sleep_model.max_depth)


print("\n" + "=" * 70)