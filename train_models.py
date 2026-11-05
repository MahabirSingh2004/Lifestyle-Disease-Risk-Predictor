# ============================================================
# LIFESTYLE DISEASE RISK PREDICTOR
# COMPLETE MODEL TRAINING PIPELINE
# ============================================================

import os
import time
import joblib
import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    brier_score_loss
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_DIR = BASE_DIR / "datasets"
MODELS_DIR = BASE_DIR / "models"

MODELS_DIR.mkdir(exist_ok=True)


# ============================================================
# DATASET PATHS
# ============================================================

SLEEP_FILE = DATASET_DIR / "Sleep_Health_Lifestyle.csv"
HEART_FILE = DATASET_DIR / "heart_disease_dataset.csv"
DIABETES_FILE = DATASET_DIR / "diabetes_pred_dataset.csv"
HEALTH_FILE = DATASET_DIR / "Health_risk_Predictor.csv"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_dataframe(df):
    """
    Basic dataset cleaning.
    """

    df = df.copy()

    # Remove spaces from column names
    df.columns = df.columns.str.strip()

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Strip string values
    for column in df.select_dtypes(include=["object"]).columns:
        df[column] = df[column].astype(str).str.strip()

    # Fill numeric missing values
    numeric_columns = df.select_dtypes(
        include=[np.number]
    ).columns

    for column in numeric_columns:
        df[column] = df[column].fillna(
            df[column].median()
        )

    # Fill categorical missing values
    categorical_columns = df.select_dtypes(
        include=["object"]
    ).columns

    for column in categorical_columns:
        if not df[column].empty:
            mode_value = df[column].mode()

            if len(mode_value) > 0:
                df[column] = df[column].fillna(
                    mode_value.iloc[0]
                )
            else:
                df[column] = df[column].fillna("Unknown")

    return df


def create_map(series):
    """
    Creates deterministic encoding map.
    """

    values = sorted(
        series.astype(str).str.strip().unique()
    )

    return {
        value: index
        for index, value in enumerate(values)
    }


def encode_column(df, column, mapping):
    """
    Applies an encoding map to a dataframe column.
    """

    df[column] = (
        df[column]
        .astype(str)
        .str.strip()
        .map(mapping)
    )

    return df


def print_separator():
    print("=" * 80)


# ============================================================
# START
# ============================================================

print_separator()
print("LOADING DATASETS")
print_separator()


# ============================================================
# LOAD DATASETS
# ============================================================

sleep_df = pd.read_csv(SLEEP_FILE)
heart_df = pd.read_csv(HEART_FILE)
diabetes_df = pd.read_csv(DIABETES_FILE)
health_df = pd.read_csv(HEALTH_FILE)


print(f"Sleep dataset    : {sleep_df.shape}")
print(f"Heart dataset    : {heart_df.shape}")
print(f"Diabetes dataset : {diabetes_df.shape}")
print(f"Health dataset   : {health_df.shape}")


# ============================================================
# CLEAN DATASETS
# ============================================================

sleep_df = clean_dataframe(sleep_df)
heart_df = clean_dataframe(heart_df)
diabetes_df = clean_dataframe(diabetes_df)
health_df = clean_dataframe(health_df)


print("\n" + "=" * 80)
print("DATA CLEANING COMPLETED")
print("=" * 80)

print(f"Sleep samples    : {len(sleep_df)}")
print(f"Heart samples    : {len(heart_df)}")
print(f"Diabetes samples : {len(diabetes_df)}")
print(f"Health samples   : {len(health_df)}")


# ============================================================
# HEALTH DATASET FEATURE ENGINEERING
# ============================================================

print("\n" + "=" * 80)
print("HEALTH DATA FEATURE ENGINEERING")
print("=" * 80)


# ------------------------------------------------------------
# BMI CATEGORY
# ------------------------------------------------------------

health_df["BMI_Category"] = pd.cut(
    health_df["bmi"],
    bins=[
        -np.inf,
        18.5,
        25,
        30,
        np.inf
    ],
    labels=[
        "Underweight",
        "Normal",
        "Overweight",
        "Obese"
    ]
)


# ------------------------------------------------------------
# SLEEP SCORE
# ------------------------------------------------------------

def calculate_sleep_score(hours):
    if 7 <= hours <= 9:
        return "Good"
    elif 6 <= hours < 7 or 9 < hours <= 10:
        return "Moderate"
    else:
        return "Poor"


health_df["Sleep_Score"] = health_df[
    "sleep_hours_per_day"
].apply(calculate_sleep_score)


# ------------------------------------------------------------
# ACTIVITY SCORE
# ------------------------------------------------------------

def calculate_activity_score(minutes):
    if minutes >= 150:
        return "High"
    elif minutes >= 75:
        return "Moderate"
    else:
        return "Low"


health_df["Activity_Score"] = health_df[
    "physical_activity_minutes_per_week"
].apply(calculate_activity_score)


# ------------------------------------------------------------
# BLOOD PRESSURE CATEGORY
# ------------------------------------------------------------

def calculate_bp_category(systolic):
    if systolic < 120:
        return "Normal"
    elif systolic < 130:
        return "Elevated"
    elif systolic < 140:
        return "High"
    else:
        return "Very High"


health_df["BP_Category"] = health_df[
    "systolic_bp"
].apply(calculate_bp_category)


# ------------------------------------------------------------
# CHOLESTEROL CATEGORY
# ------------------------------------------------------------

def calculate_cholesterol_category(value):
    if value < 200:
        return "Desirable"
    elif value < 240:
        return "Borderline High"
    else:
        return "High"


health_df["Cholesterol_Category"] = health_df[
    "cholesterol_total"
].apply(calculate_cholesterol_category)


# ------------------------------------------------------------
# LIFESTYLE SCORE
# ------------------------------------------------------------

health_df["Lifestyle_Score"] = (
    health_df["physical_activity_minutes_per_week"] / 60
    + health_df["sleep_hours_per_day"]
    - health_df["screen_time_hours_per_day"]
)


# ------------------------------------------------------------
# SAVE ENGINEERED DATA
# ------------------------------------------------------------

health_output = MODELS_DIR / "health_feature_engineered.csv"

health_df.to_csv(
    health_output,
    index=False
)

print("\nHealth feature-engineered dataset saved.")


# ============================================================
# CREATE ENCODING MAPS
# ============================================================

encoding_maps = {}


# ============================================================
# DIABETES ENCODING
# ============================================================

diabetes_gender_map = create_map(
    diabetes_df["gender"]
)

diabetes_smoking_map = create_map(
    diabetes_df["smoking_history"]
)

encoding_maps[
    "diabetes_gender"
] = diabetes_gender_map

encoding_maps[
    "diabetes_smoking_history"
] = diabetes_smoking_map


diabetes_df = encode_column(
    diabetes_df,
    "gender",
    diabetes_gender_map
)

diabetes_df = encode_column(
    diabetes_df,
    "smoking_history",
    diabetes_smoking_map
)


# ============================================================
# HEART ENCODING
# ============================================================

heart_gender_map = create_map(
    heart_df["Gender"]
)

heart_smoking_map = create_map(
    heart_df["Smoking"]
)

heart_alcohol_map = create_map(
    heart_df["Alcohol Intake"]
)

heart_family_history_map = create_map(
    heart_df["Family History"]
)

heart_diabetes_map = create_map(
    heart_df["Diabetes"]
)

heart_obesity_map = create_map(
    heart_df["Obesity"]
)

heart_angina_map = create_map(
    heart_df["Exercise Induced Angina"]
)

heart_chest_pain_map = create_map(
    heart_df["Chest Pain Type"]
)


encoding_maps[
    "heart_Gender"
] = heart_gender_map

encoding_maps[
    "heart_Smoking"
] = heart_smoking_map

encoding_maps[
    "heart_Alcohol Intake"
] = heart_alcohol_map

encoding_maps[
    "heart_Family History"
] = heart_family_history_map

encoding_maps[
    "heart_Diabetes"
] = heart_diabetes_map

encoding_maps[
    "heart_Obesity"
] = heart_obesity_map

encoding_maps[
    "heart_Exercise Induced Angina"
] = heart_angina_map

encoding_maps[
    "heart_Chest Pain Type"
] = heart_chest_pain_map


heart_df = encode_column(
    heart_df,
    "Gender",
    heart_gender_map
)

heart_df = encode_column(
    heart_df,
    "Smoking",
    heart_smoking_map
)

heart_df = encode_column(
    heart_df,
    "Alcohol Intake",
    heart_alcohol_map
)

heart_df = encode_column(
    heart_df,
    "Family History",
    heart_family_history_map
)

heart_df = encode_column(
    heart_df,
    "Diabetes",
    heart_diabetes_map
)

heart_df = encode_column(
    heart_df,
    "Obesity",
    heart_obesity_map
)

heart_df = encode_column(
    heart_df,
    "Exercise Induced Angina",
    heart_angina_map
)

heart_df = encode_column(
    heart_df,
    "Chest Pain Type",
    heart_chest_pain_map
)


# ============================================================
# SLEEP ENCODING
# ============================================================

sleep_gender_map = create_map(
    sleep_df["Gender"]
)

sleep_occupation_map = create_map(
    sleep_df["Occupation"]
)

sleep_bmi_map = create_map(
    sleep_df["BMI Category"]
)

sleep_bp_map = create_map(
    sleep_df["Blood Pressure"]
)


encoding_maps[
    "sleep_Gender"
] = sleep_gender_map

encoding_maps[
    "sleep_Occupation"
] = sleep_occupation_map

encoding_maps[
    "sleep_BMI Category"
] = sleep_bmi_map

encoding_maps[
    "sleep_Blood Pressure"
] = sleep_bp_map


sleep_df = encode_column(
    sleep_df,
    "Gender",
    sleep_gender_map
)

sleep_df = encode_column(
    sleep_df,
    "Occupation",
    sleep_occupation_map
)

sleep_df = encode_column(
    sleep_df,
    "BMI Category",
    sleep_bmi_map
)

sleep_df = encode_column(
    sleep_df,
    "Blood Pressure",
    sleep_bp_map
)


# ============================================================
# SAVE ENCODING MAPS
# ============================================================

joblib.dump(
    encoding_maps,
    MODELS_DIR / "encoding_maps.pkl"
)

print("Encoding maps saved.")


# ============================================================
# DIABETES MODEL TRAINING
# ============================================================

print("\n")
print_separator()
print("DIABETES MODEL TRAINING")
print_separator()


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


X_diabetes = diabetes_df[
    diabetes_features
]

y_diabetes = diabetes_df[
    "diabetes"
]


# ------------------------------------------------------------
# TRAIN TEST SPLIT
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_diabetes,
    y_diabetes,
    test_size=0.20,
    random_state=42,
    stratify=y_diabetes
)


# ------------------------------------------------------------
# SCALER
# ------------------------------------------------------------

diabetes_scaler = StandardScaler()

X_train_scaled = diabetes_scaler.fit_transform(
    X_train
)

X_test_scaled = diabetes_scaler.transform(
    X_test
)


joblib.dump(
    diabetes_scaler,
    MODELS_DIR / "diabetes_scaler.pkl"
)


# ------------------------------------------------------------
# CANDIDATE MODELS
# ------------------------------------------------------------

diabetes_candidates = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=1000,
            random_state=42
        ),

    "Decision Tree":
        DecisionTreeClassifier(
            max_depth=10,
            random_state=42
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        ),

    "Naive Bayes":
        GaussianNB()
}


diabetes_results = {}


# ------------------------------------------------------------
# TRAIN ALL DIABETES MODELS
# ------------------------------------------------------------

for model_name, model in diabetes_candidates.items():

    start_time = time.time()

    model.fit(
        X_train_scaled,
        y_train
    )

    predictions = model.predict(
        X_test_scaled
    )

    probabilities = model.predict_proba(
        X_test_scaled
    )[:, 1]

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    log_loss_value = log_loss(
        y_test,
        probabilities
    )

    brier_value = brier_score_loss(
        y_test,
        probabilities
    )

    training_time = time.time() - start_time

    diabetes_results[
        model_name
    ] = {
        "model": model,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "log_loss": log_loss_value,
        "brier": brier_value,
        "training_time": training_time
    }

    print(f"\n{model_name}")
    print(f"Accuracy    : {accuracy:.4f}")
    print(f"Precision   : {precision:.4f}")
    print(f"Recall      : {recall:.4f}")
    print(f"F1          : {f1:.4f}")
    print(f"ROC_AUC     : {roc_auc:.4f}")
    print(f"Log_Loss    : {log_loss_value:.4f}")
    print(f"Brier       : {brier_value:.4f}")
    print(f"Train Time  : {training_time:.2f}s")


# ============================================================
# DIABETES MODEL SELECTION
# ============================================================
#
# IMPORTANT:
#
# This project is a RISK prediction system.
#
# Therefore probability quality is important.
#
# Lower Log Loss  = better
# Lower Brier     = better
# Higher ROC AUC  = better
#
# We first prioritize Log Loss,
# then Brier Score,
# then ROC AUC.
# ============================================================

selected_diabetes_name = min(
    diabetes_results,
    key=lambda name: (
        diabetes_results[name]["log_loss"],
        diabetes_results[name]["brier"],
        -diabetes_results[name]["roc_auc"]
    )
)


selected_diabetes_model = diabetes_results[
    selected_diabetes_name
]["model"]


print(
    f"\nSelected Diabetes: "
    f"{selected_diabetes_name}"
)


# ------------------------------------------------------------
# CALIBRATE DIABETES MODEL
# ------------------------------------------------------------

calibrated_diabetes_model = CalibratedClassifierCV(
    estimator=selected_diabetes_model,
    method="sigmoid",
    cv=5
)


calibrated_diabetes_model.fit(
    X_train_scaled,
    y_train
)


# ------------------------------------------------------------
# SAVE DIABETES MODEL
# ------------------------------------------------------------

joblib.dump(
    calibrated_diabetes_model,
    MODELS_DIR / "diabetes_model.pkl"
)


# ============================================================
# HEART DISEASE MODEL TRAINING
# ============================================================

print("\n")
print_separator()
print("HEART DISEASE MODEL TRAINING")
print_separator()


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


X_heart = heart_df[
    heart_features
]

y_heart = heart_df[
    "Heart Disease"
]


# ------------------------------------------------------------
# TRAIN TEST SPLIT
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_heart,
    y_heart,
    test_size=0.20,
    random_state=42,
    stratify=y_heart
)


# ------------------------------------------------------------
# SCALER
# ------------------------------------------------------------

heart_scaler = StandardScaler()

X_train_scaled = heart_scaler.fit_transform(
    X_train
)

X_test_scaled = heart_scaler.transform(
    X_test
)


joblib.dump(
    heart_scaler,
    MODELS_DIR / "heart_scaler.pkl"
)


# ============================================================
# HEART MODEL
# ============================================================
#
# Logistic Regression is deliberately used because:
#
# - ROC-AUC was strong
# - probabilities are smoother
# - easier to interpret
# - suitable for risk prediction
# ============================================================

heart_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)


heart_model.fit(
    X_train_scaled,
    y_train
)


heart_predictions = heart_model.predict(
    X_test_scaled
)

heart_probabilities = heart_model.predict_proba(
    X_test_scaled
)[:, 1]


heart_accuracy = accuracy_score(
    y_test,
    heart_predictions
)

heart_precision = precision_score(
    y_test,
    heart_predictions,
    zero_division=0
)

heart_recall = recall_score(
    y_test,
    heart_predictions,
    zero_division=0
)

heart_f1 = f1_score(
    y_test,
    heart_predictions,
    zero_division=0
)

heart_roc_auc = roc_auc_score(
    y_test,
    heart_probabilities
)

heart_log_loss = log_loss(
    y_test,
    heart_probabilities
)

heart_brier = brier_score_loss(
    y_test,
    heart_probabilities
)


print("\nLogistic Regression")
print(f"Accuracy    : {heart_accuracy:.4f}")
print(f"Precision   : {heart_precision:.4f}")
print(f"Recall      : {heart_recall:.4f}")
print(f"F1          : {heart_f1:.4f}")
print(f"ROC_AUC     : {heart_roc_auc:.4f}")
print(f"Log_Loss    : {heart_log_loss:.4f}")
print(f"Brier       : {heart_brier:.4f}")


print("\nSelected Heart: Logistic Regression")


# ------------------------------------------------------------
# CALIBRATE HEART MODEL
# ------------------------------------------------------------

calibrated_heart_model = CalibratedClassifierCV(
    estimator=heart_model,
    method="sigmoid",
    cv=5
)


calibrated_heart_model.fit(
    X_train_scaled,
    y_train
)


# ------------------------------------------------------------
# SAVE HEART MODEL
# ------------------------------------------------------------

joblib.dump(
    calibrated_heart_model,
    MODELS_DIR / "heart_model.pkl"
)


# ============================================================
# SLEEP DISORDER MODEL TRAINING
# ============================================================

print("\n")
print_separator()
print("SLEEP DISORDER MODEL TRAINING")
print_separator()


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


X_sleep = sleep_df[
    sleep_features
]

y_sleep = sleep_df[
    "Sleep Disorder"
]


# ------------------------------------------------------------
# TRAIN TEST SPLIT
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_sleep,
    y_sleep,
    test_size=0.20,
    random_state=42,
    stratify=y_sleep
)


# ------------------------------------------------------------
# SCALER
# ------------------------------------------------------------

sleep_scaler = StandardScaler()

X_train_scaled = sleep_scaler.fit_transform(
    X_train
)

X_test_scaled = sleep_scaler.transform(
    X_test
)


joblib.dump(
    sleep_scaler,
    MODELS_DIR / "sleep_scaler.pkl"
)


# ------------------------------------------------------------
# SLEEP CANDIDATE MODELS
# ------------------------------------------------------------

sleep_candidates = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=1000,
            random_state=42
        ),

    "Decision Tree":
        DecisionTreeClassifier(
            max_depth=10,
            random_state=42
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        ),

    "Naive Bayes":
        GaussianNB()
}


sleep_results = {}


# ------------------------------------------------------------
# TRAIN SLEEP MODELS
# ------------------------------------------------------------

for model_name, model in sleep_candidates.items():

    start_time = time.time()

    model.fit(
        X_train_scaled,
        y_train
    )

    predictions = model.predict(
        X_test_scaled
    )

    probabilities = model.predict_proba(
        X_test_scaled
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    log_loss_value = log_loss(
        y_test,
        probabilities
    )

    training_time = time.time() - start_time

    sleep_results[
        model_name
    ] = {
        "model": model,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "log_loss": log_loss_value,
        "training_time": training_time
    }

    print(f"\n{model_name}")
    print(f"Accuracy    : {accuracy:.4f}")
    print(f"Precision   : {precision:.4f}")
    print(f"Recall      : {recall:.4f}")
    print(f"F1          : {f1:.4f}")
    print(f"Log_Loss    : {log_loss_value:.4f}")
    print(f"Train Time  : {training_time:.2f}s")


# ------------------------------------------------------------
# SELECT SLEEP MODEL
# ------------------------------------------------------------

selected_sleep_name = max(
    sleep_results,
    key=lambda name: (
        sleep_results[name]["f1"],
        sleep_results[name]["accuracy"],
        -sleep_results[name]["log_loss"]
    )
)


selected_sleep_model = sleep_results[
    selected_sleep_name
]["model"]


print(
    f"\nSelected Sleep: "
    f"{selected_sleep_name}"
)


# ------------------------------------------------------------
# CALIBRATE SLEEP MODEL
# ------------------------------------------------------------

calibrated_sleep_model = CalibratedClassifierCV(
    estimator=selected_sleep_model,
    method="sigmoid",
    cv=5
)


calibrated_sleep_model.fit(
    X_train_scaled,
    y_train
)


# ------------------------------------------------------------
# SAVE SLEEP MODEL
# ------------------------------------------------------------

joblib.dump(
    calibrated_sleep_model,
    MODELS_DIR / "sleep_model.pkl"
)


# ============================================================
# MODEL METADATA
# ============================================================

metadata = {

    "diabetes": {
        "selected_model": selected_diabetes_name,
        "features": diabetes_features,
        "probability_calibrated": True,
        "calibration_method": "sigmoid"
    },

    "heart": {
        "selected_model": "Logistic Regression",
        "features": heart_features,
        "probability_calibrated": True,
        "calibration_method": "sigmoid"
    },

    "sleep": {
        "selected_model": selected_sleep_name,
        "features": sleep_features,
        "probability_calibrated": True,
        "calibration_method": "sigmoid"
    }
}


joblib.dump(
    metadata,
    MODELS_DIR / "model_metadata.pkl"
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n")
print_separator()
print("TRAINING COMPLETED SUCCESSFULLY")
print_separator()

print("\nSaved files:")

print("[OK] diabetes_model.pkl")
print("[OK] diabetes_scaler.pkl")

print("[OK] heart_model.pkl")
print("[OK] heart_scaler.pkl")

print("[OK] sleep_model.pkl")
print("[OK] sleep_scaler.pkl")

print("[OK] encoding_maps.pkl")
print("[OK] model_metadata.pkl")

print("[OK] health_feature_engineered.csv")


print("\n")
print_separator()
print("MODEL SELECTION")
print_separator()

print(
    f"Diabetes       : "
    f"{selected_diabetes_name}"
)

print(
    "Heart Disease  : Logistic Regression"
)

print(
    f"Sleep Disorder : "
    f"{selected_sleep_name}"
)

print_separator()