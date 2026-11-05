
# ============================================================
# MODEL EVALUATION
# Lifestyle Disease Risk Predictor + Habit Simulator
#
# Evaluates existing saved models without retraining them.
# ============================================================

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    brier_score_loss,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

from sklearn.preprocessing import label_binarize
from sklearn.calibration import calibration_curve


warnings.filterwarnings("ignore")


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_DIR = BASE_DIR / "datasets"
MODEL_DIR = BASE_DIR / "models"

REPORT_DIR = BASE_DIR / "model_evaluation"

REPORT_DIR.mkdir(exist_ok=True)


# ============================================================
# DATASET PATHS
# ============================================================

DIABETES_FILE = DATASET_DIR / "diabetes_pred_dataset.csv"
HEART_FILE = DATASET_DIR / "heart_disease_dataset.csv"
SLEEP_FILE = DATASET_DIR / "Sleep_Health_Lifestyle.csv"


# ============================================================
# LOAD SAVED MODELS, SCALERS AND ENCODING MAPS
# ============================================================

print("=" * 75)
print("LOADING SAVED MODELS")
print("=" * 75)

encoding_maps = joblib.load(
    MODEL_DIR / "encoding_maps.pkl"
)

diabetes_model = joblib.load(
    MODEL_DIR / "diabetes_model.pkl"
)

diabetes_scaler = joblib.load(
    MODEL_DIR / "diabetes_scaler.pkl"
)

heart_model = joblib.load(
    MODEL_DIR / "heart_model.pkl"
)

heart_scaler = joblib.load(
    MODEL_DIR / "heart_scaler.pkl"
)

sleep_model = joblib.load(
    MODEL_DIR / "sleep_model.pkl"
)

sleep_scaler = joblib.load(
    MODEL_DIR / "sleep_scaler.pkl"
)

print("[OK] All saved models loaded.")
print("[OK] All saved scalers loaded.")
print("[OK] Encoding maps loaded.")


# ============================================================
# DATA CLEANING
# REPLICATES THE TRAINING CODE
# ============================================================

def clean_dataframe(df):

    df = df.copy()

    df.columns = df.columns.str.strip()

    df = df.drop_duplicates()

    for column in df.select_dtypes(
        include=["object"]
    ).columns:

        df[column] = (
            df[column]
            .astype(str)
            .str.strip()
        )

    numeric_columns = df.select_dtypes(
        include=[np.number]
    ).columns

    for column in numeric_columns:

        df[column] = df[column].fillna(
            df[column].median()
        )

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

                df[column] = df[column].fillna(
                    "Unknown"
                )

    return df


# ============================================================
# APPLY EXISTING ENCODING MAPS
# ============================================================

def apply_encoding(df, column, mapping):

    df[column] = (
        df[column]
        .astype(str)
        .str.strip()
        .map(mapping)
    )

    if df[column].isna().any():

        raise ValueError(
            f"Unknown categorical value found in "
            f"column '{column}'. Check the saved encoding map."
        )

    return df


# ============================================================
# EVALUATION AND VISUALIZATION FUNCTION
# ============================================================

def evaluate_model(
    model_name,
    model,
    scaler,
    X,
    y,
    test_size=0.20,
    random_state=42
):

    print("\n")
    print("=" * 75)
    print(f"{model_name.upper()} MODEL EVALUATION")
    print("=" * 75)

    # --------------------------------------------------------
    # RECREATE THE SAME TRAIN-TEST SPLIT
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print(f"Total samples : {len(X)}")
    print(f"Training data : {len(X_train)}")
    print(f"Testing data  : {len(X_test)}")

    # --------------------------------------------------------
    # APPLY EXISTING SCALER
    # --------------------------------------------------------

    X_test_scaled = scaler.transform(X_test)

    # --------------------------------------------------------
    # PREDICTIONS FROM SAVED MODEL
    # --------------------------------------------------------

    predictions = model.predict(
        X_test_scaled
    )

    probabilities = model.predict_proba(
        X_test_scaled
    )

    classes = np.asarray(model.classes_)

    if len(classes) < 2:
        raise ValueError(
            f"{model_name} has fewer than two model classes."
        )

    # Ensure labels and model classes are compatible.
    if not set(pd.unique(y_test)).issubset(set(classes)):
        raise ValueError(
            f"Test labels do not match the saved model classes "
            f"for {model_name}."
        )

    # --------------------------------------------------------
    # COMMON METRICS
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    if len(classes) == 2:

        # Use class 1 as positive when available.
        positive_class = (
            1 if 1 in classes else classes[-1]
        )

        precision = precision_score(
            y_test,
            predictions,
            pos_label=positive_class,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            pos_label=positive_class,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            predictions,
            pos_label=positive_class,
            zero_division=0
        )

        positive_index = list(classes).index(
            positive_class
        )

        y_binary = (
            np.asarray(y_test) == positive_class
        ).astype(int)

        positive_probabilities = probabilities[
            :, positive_index
        ]

        roc_auc = roc_auc_score(
            y_binary,
            positive_probabilities
        )

        log_loss_value = log_loss(
            y_test,
            probabilities,
            labels=classes
        )

        brier_score = brier_score_loss(
            y_binary,
            positive_probabilities
        )

        average_type = "Binary"

    else:

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

        roc_auc = roc_auc_score(
            y_test,
            probabilities,
            labels=classes,
            multi_class="ovr",
            average="weighted"
        )

        log_loss_value = log_loss(
            y_test,
            probabilities,
            labels=classes
        )

        # Multiclass Brier score:
        # Mean sum of squared differences between
        # one-hot actual labels and predicted probabilities.

        actual_one_hot = label_binarize(
            y_test,
            classes=classes
        )

        brier_score = np.mean(
            np.sum(
                (actual_one_hot - probabilities) ** 2,
                axis=1
            )
        )

        average_type = "Weighted"

    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print("\nMODEL PERFORMANCE")
    print("-" * 50)

    print(f"Accuracy    : {accuracy:.4f}")
    print(f"Precision   : {precision:.4f}")
    print(f"Recall      : {recall:.4f}")
    print(f"F1-score    : {f1:.4f}")
    print(f"ROC-AUC     : {roc_auc:.4f}")
    print(f"Log Loss    : {log_loss_value:.4f}")
    print(f"Brier Score : {brier_score:.4f}")

    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    report = classification_report(
        y_test,
        predictions,
        labels=classes,
        target_names=[str(c) for c in classes],
        zero_division=0
    )

    print("\nCLASSIFICATION REPORT")
    print("-" * 50)
    print(report)

    report_file = REPORT_DIR / (
        model_name.lower().replace(" ", "_")
        + "_classification_report.txt"
    )

    with open(report_file, "w", encoding="utf-8") as file:

        file.write(
            f"{model_name} Classification Report\n"
        )

        file.write("=" * 60 + "\n\n")

        file.write(report)

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=classes
    )

    plt.figure(figsize=(7, 5))

    plt.imshow(cm, interpolation="nearest")

    plt.title(
        f"{model_name} - Confusion Matrix"
    )

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(
        tick_marks,
        [str(c) for c in classes],
        rotation=30
    )

    plt.yticks(
        tick_marks,
        [str(c) for c in classes]
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")

    plt.tight_layout()

    for i in range(cm.shape[0]):

        for j in range(cm.shape[1]):

            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center"
            )

    cm_file = REPORT_DIR / (
        model_name.lower().replace(" ", "_")
        + "_confusion_matrix.png"
    )

    plt.savefig(
        cm_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # ROC CURVE
    # --------------------------------------------------------

    plt.figure(figsize=(7, 5))

    if len(classes) == 2:

        fpr, tpr, _ = roc_curve(
            y_binary,
            positive_probabilities
        )

        plt.plot(
            fpr,
            tpr,
            label=f"ROC-AUC = {roc_auc:.4f}"
        )

    else:

        y_test_binary = label_binarize(
            y_test,
            classes=classes
        )

        for i, class_name in enumerate(classes):

            if len(np.unique(y_test_binary[:, i])) < 2:
                continue

            fpr, tpr, _ = roc_curve(
                y_test_binary[:, i],
                probabilities[:, i]
            )

            class_auc = auc(
                fpr,
                tpr
            )

            plt.plot(
                fpr,
                tpr,
                label=f"{class_name} (AUC = {class_auc:.3f})"
            )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random Classifier"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.title(
        f"{model_name} - ROC Curve"
    )

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    roc_file = REPORT_DIR / (
        model_name.lower().replace(" ", "_")
        + "_roc_curve.png"
    )

    plt.savefig(
        roc_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # PROBABILITY CALIBRATION CURVE
    # Binary classification only
    # --------------------------------------------------------

    if len(classes) == 2:

        fraction_positive, mean_probability = calibration_curve(
            y_binary,
            positive_probabilities,
            n_bins=10,
            strategy="uniform"
        )

        plt.figure(figsize=(7, 5))

        plt.plot(
            mean_probability,
            fraction_positive,
            marker="o",
            label="Saved Model"
        )

        plt.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            label="Perfect Calibration"
        )

        plt.xlabel("Mean Predicted Probability")
        plt.ylabel("Fraction of Positive Cases")

        plt.title(
            f"{model_name} - Calibration Curve"
        )

        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        calibration_file = REPORT_DIR / (
            model_name.lower().replace(" ", "_")
            + "_calibration_curve.png"
        )

        plt.savefig(
            calibration_file,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

    # --------------------------------------------------------
    # SAVE CONFUSION MATRIX DATA
    # --------------------------------------------------------

    cm_df = pd.DataFrame(
        cm,
        index=[f"Actual_{c}" for c in classes],
        columns=[f"Predicted_{c}" for c in classes]
    )

    cm_df.to_csv(
        REPORT_DIR / (
            model_name.lower().replace(" ", "_")
            + "_confusion_matrix.csv"
        )
    )

    # --------------------------------------------------------
    # RETURN METRICS
    # --------------------------------------------------------

    result = {
        "Model": model_name,
        "Test Samples": len(y_test),
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC AUC": roc_auc,
        "Log Loss": log_loss_value,
        "Brier Score": brier_score,
        "Average Type": average_type
    }

    print("\n[OK] Evaluation completed.")

    return result


# ============================================================
# DIABETES DATASET PREPARATION
# ============================================================

print("\n")
print("=" * 75)
print("PREPARING DIABETES DATASET")
print("=" * 75)

diabetes_df = pd.read_csv(
    DIABETES_FILE
)

diabetes_df = clean_dataframe(
    diabetes_df
)

diabetes_df = apply_encoding(
    diabetes_df,
    "gender",
    encoding_maps["diabetes_gender"]
)

diabetes_df = apply_encoding(
    diabetes_df,
    "smoking_history",
    encoding_maps["diabetes_smoking_history"]
)

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


# ============================================================
# HEART DISEASE DATASET PREPARATION
# ============================================================

print("\n")
print("=" * 75)
print("PREPARING HEART DISEASE DATASET")
print("=" * 75)

heart_df = pd.read_csv(
    HEART_FILE
)

heart_df = clean_dataframe(
    heart_df
)

heart_encoding_columns = {
    "Gender": "heart_Gender",
    "Smoking": "heart_Smoking",
    "Alcohol Intake": "heart_Alcohol Intake",
    "Family History": "heart_Family History",
    "Diabetes": "heart_Diabetes",
    "Obesity": "heart_Obesity",
    "Exercise Induced Angina": "heart_Exercise Induced Angina",
    "Chest Pain Type": "heart_Chest Pain Type"
}

for column, map_name in heart_encoding_columns.items():

    heart_df = apply_encoding(
        heart_df,
        column,
        encoding_maps[map_name]
    )

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


# ============================================================
# SLEEP DISORDER DATASET PREPARATION
# ============================================================

print("\n")
print("=" * 75)
print("PREPARING SLEEP DISORDER DATASET")
print("=" * 75)

sleep_df = pd.read_csv(
    SLEEP_FILE
)

sleep_df = clean_dataframe(
    sleep_df
)

sleep_encoding_columns = {
    "Gender": "sleep_Gender",
    "Occupation": "sleep_Occupation",
    "BMI Category": "sleep_BMI Category",
    "Blood Pressure": "sleep_Blood Pressure"
}

for column, map_name in sleep_encoding_columns.items():

    sleep_df = apply_encoding(
        sleep_df,
        column,
        encoding_maps[map_name]
    )

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


# ============================================================
# EVALUATE ALL THREE SAVED MODELS
# ============================================================

all_results = []

# ------------------------------------------------------------
# 1. DIABETES
# ------------------------------------------------------------

diabetes_result = evaluate_model(
    model_name="Diabetes",
    model=diabetes_model,
    scaler=diabetes_scaler,
    X=X_diabetes,
    y=y_diabetes
)

all_results.append(
    diabetes_result
)


# ------------------------------------------------------------
# 2. HEART DISEASE
# ------------------------------------------------------------

heart_result = evaluate_model(
    model_name="Heart Disease",
    model=heart_model,
    scaler=heart_scaler,
    X=X_heart,
    y=y_heart
)

all_results.append(
    heart_result
)


# ------------------------------------------------------------
# 3. SLEEP DISORDER
# ------------------------------------------------------------

sleep_result = evaluate_model(
    model_name="Sleep Disorder",
    model=sleep_model,
    scaler=sleep_scaler,
    X=X_sleep,
    y=y_sleep
)

all_results.append(
    sleep_result
)


# ============================================================
# FINAL COMPARISON REPORT
# ============================================================

print("\n")
print("=" * 75)
print("FINAL MODEL COMPARISON")
print("=" * 75)

results_df = pd.DataFrame(
    all_results
)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

results_file = REPORT_DIR / "model_evaluation_summary.csv"

results_df.to_csv(
    results_file,
    index=False
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n")
print("=" * 75)
print("MODEL EVALUATION COMPLETED SUCCESSFULLY")
print("=" * 75)

print("\nGenerated files:")

print("[OK] model_evaluation_summary.csv")

print("[OK] diabetes_classification_report.txt")
print("[OK] diabetes_confusion_matrix.png")
print("[OK] diabetes_roc_curve.png")
print("[OK] diabetes_calibration_curve.png")

print("[OK] heart_disease_classification_report.txt")
print("[OK] heart_disease_confusion_matrix.png")
print("[OK] heart_disease_roc_curve.png")
print("[OK] heart_disease_calibration_curve.png")

print("[OK] sleep_disorder_classification_report.txt")
print("[OK] sleep_disorder_confusion_matrix.png")
print("[OK] sleep_disorder_roc_curve.png")

print(f"\nReport folder: {REPORT_DIR}")

print("=" * 75)
