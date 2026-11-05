# ============================================================
# PREDICTION MODULE
# ============================================================

import os
import joblib
import pandas as pd


# ============================================================
# BASE DIRECTORIES
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)


# ============================================================
# LOAD ENCODING MAPS
# ============================================================

encoding_maps = joblib.load(
    os.path.join(
        MODEL_DIR,
        "encoding_maps.pkl"
    )
)


# ============================================================
# DIABETES MODEL
# ============================================================

diabetes_model = joblib.load(
    os.path.join(
        MODEL_DIR,
        "diabetes_model.pkl"
    )
)

diabetes_scaler = joblib.load(
    os.path.join(
        MODEL_DIR,
        "diabetes_scaler.pkl"
    )
)


def predict_diabetes(
    gender,
    age,
    hypertension,
    heart_disease,
    smoking_history,
    bmi,
    hba1c_level,
    blood_glucose_level
):

    gender_value = encoding_maps[
        "diabetes_gender"
    ][gender]

    smoking_value = encoding_maps[
        "diabetes_smoking_history"
    ][smoking_history]

    input_data = pd.DataFrame(
        [[
            gender_value,
            age,
            hypertension,
            heart_disease,
            smoking_value,
            bmi,
            hba1c_level,
            blood_glucose_level
        ]],
        columns=[
            "gender",
            "age",
            "hypertension",
            "heart_disease",
            "smoking_history",
            "bmi",
            "HbA1c_level",
            "blood_glucose_level"
        ]
    )

    scaled_input = diabetes_scaler.transform(
        input_data
    )

    prediction = diabetes_model.predict(
        scaled_input
    )[0]

    probabilities = diabetes_model.predict_proba(
        scaled_input
    )[0]

    classes = list(
        diabetes_model.classes_
    )

    positive_index = classes.index(1)

    probability = probabilities[
        positive_index
    ]

    return {
        "prediction": int(prediction),
        "probability": float(probability)
    }


# ============================================================
# HEART DISEASE MODEL
# ============================================================

heart_model = joblib.load(
    os.path.join(
        MODEL_DIR,
        "heart_model.pkl"
    )
)

heart_scaler = joblib.load(
    os.path.join(
        MODEL_DIR,
        "heart_scaler.pkl"
    )
)


def predict_heart_disease(
    age,
    gender,
    cholesterol,
    blood_pressure,
    heart_rate,
    smoking,
    alcohol_intake,
    exercise_hours,
    family_history,
    diabetes,
    obesity,
    stress_level,
    blood_sugar,
    exercise_induced_angina,
    chest_pain_type
):

    # --------------------------------------------------------
    # Convert categorical values using ACTUAL map keys
    # --------------------------------------------------------

    gender_value = encoding_maps[
        "heart_Gender"
    ][gender]

    smoking_value = encoding_maps[
        "heart_Smoking"
    ][smoking]

    alcohol_value = encoding_maps[
        "heart_Alcohol Intake"
    ][alcohol_intake]

    family_history_value = encoding_maps[
        "heart_Family History"
    ][family_history]

    diabetes_value = encoding_maps[
        "heart_Diabetes"
    ][diabetes]

    obesity_value = encoding_maps[
        "heart_Obesity"
    ][obesity]

    angina_value = encoding_maps[
        "heart_Exercise Induced Angina"
    ][exercise_induced_angina]

    chest_pain_value = encoding_maps[
        "heart_Chest Pain Type"
    ][chest_pain_type]

    # --------------------------------------------------------
    # Create input DataFrame
    # --------------------------------------------------------

    input_data = pd.DataFrame(
        [[
            age,
            gender_value,
            cholesterol,
            blood_pressure,
            heart_rate,
            smoking_value,
            alcohol_value,
            exercise_hours,
            family_history_value,
            diabetes_value,
            obesity_value,
            stress_level,
            blood_sugar,
            angina_value,
            chest_pain_value
        ]],
        columns=[
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
    )

    # --------------------------------------------------------
    # Scale input
    # --------------------------------------------------------

    scaled_input = heart_scaler.transform(
        input_data
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction = heart_model.predict(
        scaled_input
    )[0]

    probabilities = heart_model.predict_proba(
        scaled_input
    )[0]

    classes = list(
        heart_model.classes_
    )

    positive_index = classes.index(1)

    probability = probabilities[
        positive_index
    ]

    return {
        "prediction": int(prediction),
        "probability": float(probability)
    }


# ============================================================
# SLEEP DISORDER MODEL
# ============================================================

sleep_model = joblib.load(
    os.path.join(
        MODEL_DIR,
        "sleep_model.pkl"
    )
)

sleep_scaler = joblib.load(
    os.path.join(
        MODEL_DIR,
        "sleep_scaler.pkl"
    )
)


def predict_sleep_disorder(
    person_id,
    gender,
    age,
    occupation,
    sleep_duration,
    quality_of_sleep,
    physical_activity_level,
    stress_level,
    bmi_category,
    blood_pressure,
    heart_rate,
    daily_steps
):

    # --------------------------------------------------------
    # Person ID is NOT used by the model.
    #
    # It remains in the function arguments so the existing
    # frontend can continue passing Person ID.
    # --------------------------------------------------------

    gender_value = encoding_maps[
        "sleep_Gender"
    ][gender]

    occupation_value = encoding_maps[
        "sleep_Occupation"
    ][occupation]

    bmi_value = encoding_maps[
        "sleep_BMI Category"
    ][bmi_category]

    bp_value = encoding_maps[
        "sleep_Blood Pressure"
    ][blood_pressure]

    # --------------------------------------------------------
    # Create input DataFrame
    # --------------------------------------------------------

    input_data = pd.DataFrame(
        [[
            gender_value,
            age,
            occupation_value,
            sleep_duration,
            quality_of_sleep,
            physical_activity_level,
            stress_level,
            bmi_value,
            bp_value,
            heart_rate,
            daily_steps
        ]],
        columns=[
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
    )

    # --------------------------------------------------------
    # Scale input
    # --------------------------------------------------------

    scaled_input = sleep_scaler.transform(
        input_data
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction = sleep_model.predict(
        scaled_input
    )[0]

    probabilities = sleep_model.predict_proba(
        scaled_input
    )[0]

    classes = list(
        sleep_model.classes_
    )

    # --------------------------------------------------------
    # Sleep Apnea probability
    #
    # The saved model classes are:
    # ['Insomnia', 'Sleep Apnea']
    #
    # We specifically return Sleep Apnea probability.
    # --------------------------------------------------------

    if "Sleep Apnea" in classes:

        positive_index = classes.index(
            "Sleep Apnea"
        )

        probability = probabilities[
            positive_index
        ]

    else:

        predicted_index = classes.index(
            prediction
        )

        probability = probabilities[
            predicted_index
        ]

    return {
        "prediction": str(prediction),
        "probability": float(probability)
    }