import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from prediction.prediction import (
    diabetes_model,
    diabetes_scaler,
    heart_model,
    heart_scaler,
    sleep_model,
    sleep_scaler,
    encoding_maps
)


# ============================================================
# COMMON SHAP EXPLANATION FUNCTION
# ============================================================

def generate_shap_explanation(
    model,
    scaler,
    input_data,
    positive_class,
    max_evals=None
):
    """
    Generate a local SHAP explanation for one prediction.

    SHAP is calculated in standardized feature space because
    the trained models receive scaled input.

    The reference point is a zero vector in standardized
    space. With the saved StandardScaler, this corresponds
    to the training-feature means.

    SHAP values explain the model's predicted probability
    for the selected positive class.

    Returns:
        SHAP values
        Base probability
        Predicted probability
        Feature names and values
        Ranked feature contributions
        SHAP Explanation object
    """

    if not isinstance(input_data, pd.DataFrame):
        raise TypeError("input_data must be a pandas DataFrame.")

    if len(input_data) != 1:
        raise ValueError(
            "This function currently explains one prediction at a time."
        )

    # --------------------------------------------------------
    # 1. Scale input using the existing saved scaler
    # --------------------------------------------------------

    scaled_input = scaler.transform(input_data)

    # --------------------------------------------------------
    # 2. Identify the selected positive class
    # --------------------------------------------------------

    classes = list(model.classes_)

    if positive_class not in classes:
        raise ValueError(
            f"Target class '{positive_class}' not found. "
            f"Available classes: {classes}"
        )

    positive_index = classes.index(positive_class)

    # --------------------------------------------------------
    # 3. Prediction function for SHAP
    # --------------------------------------------------------

    def predict_probability(data):

        data = np.asarray(data)

        probabilities = model.predict_proba(data)

        return probabilities[:, positive_index]

    # --------------------------------------------------------
    # 4. Reference baseline
    # --------------------------------------------------------

    background = np.zeros(
        (1, scaled_input.shape[1])
    )

    masker = shap.maskers.Independent(background)

    explainer = shap.PermutationExplainer(
        predict_probability,
        masker
    )

    number_of_features = scaled_input.shape[1]

    if max_evals is None:
        max_evals = 10 * (
            2 * number_of_features + 1
        )

    # --------------------------------------------------------
    # 5. Generate SHAP explanation
    # --------------------------------------------------------

    raw_explanation = explainer(
        scaled_input,
        max_evals=max_evals
    )

    shap_values = np.asarray(
        raw_explanation.values
    )

    # Handle possible output dimensions
    if shap_values.ndim == 3:
        shap_values = shap_values[0, :, 0]

    elif shap_values.ndim == 2:
        shap_values = shap_values[0]

    else:
        shap_values = shap_values.reshape(-1)

    # --------------------------------------------------------
    # 6. Extract baseline probability
    # --------------------------------------------------------

    base_value = float(
        np.asarray(
            raw_explanation.base_values
        ).reshape(-1)[0]
    )

    # --------------------------------------------------------
    # 7. Get actual model probability
    # --------------------------------------------------------

    predicted_probability = float(
        predict_probability(scaled_input)[0]
    )

    # --------------------------------------------------------
    # 8. Feature information
    # --------------------------------------------------------

    feature_names = list(input_data.columns)

    # Original encoded values supplied to the scaler
    feature_values = input_data.iloc[0].to_dict()

    # Values actually received by the model
    scaled_feature_values = scaled_input[0].tolist()

    # --------------------------------------------------------
    # 9. Create correctly aligned SHAP Explanation
    # --------------------------------------------------------

    explanation = shap.Explanation(
        values=shap_values,
        base_values=base_value,

        # SHAP values were calculated on scaled values.
        data=scaled_input[0],

        # Keep original encoded values available for display.
        display_data=input_data.iloc[0].to_numpy(),

        feature_names=feature_names
    )

    # --------------------------------------------------------
    # 10. Rank feature contributions
    # --------------------------------------------------------

    feature_contributions = []

    for index, feature in enumerate(feature_names):

        contribution = float(shap_values[index])

        if contribution > 0:
            direction = "Increases predicted probability"

        elif contribution < 0:
            direction = "Decreases predicted probability"

        else:
            direction = "No meaningful contribution"

        feature_contributions.append({
            "feature": feature,
            "input_value": feature_values[feature],
            "shap_value": contribution,
            "absolute_shap": abs(contribution),
            "direction": direction
        })

    feature_contributions.sort(
        key=lambda item: item["absolute_shap"],
        reverse=True
    )

    # --------------------------------------------------------
    # 11. Check SHAP additive property
    # --------------------------------------------------------

    reconstructed_probability = (
        base_value + float(np.sum(shap_values))
    )

    additivity_difference = abs(
        predicted_probability - reconstructed_probability
    )

    return {
        "shap_values": shap_values,
        "base_value": base_value,
        "prediction_probability": predicted_probability,
        "feature_names": feature_names,
        "feature_values": feature_values,
        "scaled_feature_values": scaled_feature_values,
        "feature_contributions": feature_contributions,
        "explanation": explanation,
        "positive_class": positive_class,
        "reconstructed_probability": reconstructed_probability,
        "additivity_difference": additivity_difference
    }


# ============================================================
# 1. DIABETES SHAP EXPLANATION
# ============================================================

def explain_diabetes(
    gender,
    age,
    hypertension,
    heart_disease,
    smoking_history,
    bmi,
    hba1c_level,
    blood_glucose_level,
    max_evals=None
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

    return generate_shap_explanation(
        model=diabetes_model,
        scaler=diabetes_scaler,
        input_data=input_data,
        positive_class=1,
        max_evals=max_evals
    )


# ============================================================
# 2. HEART DISEASE SHAP EXPLANATION
# ============================================================

def explain_heart_disease(
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
    chest_pain_type,
    max_evals=None
):

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

    return generate_shap_explanation(
        model=heart_model,
        scaler=heart_scaler,
        input_data=input_data,
        positive_class=1,
        max_evals=max_evals
    )


# ============================================================
# 3. SLEEP DISORDER SHAP EXPLANATION
# ============================================================

def explain_sleep_disorder(
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
    daily_steps,
    max_evals=None
):

    # Person ID is not used by the trained model.

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

    return generate_shap_explanation(
        model=sleep_model,
        scaler=sleep_scaler,
        input_data=input_data,
        positive_class="Sleep Apnea",
        max_evals=max_evals
    )


# ============================================================
# 4. LOCAL SHAP WATERFALL PLOT
# ============================================================

def create_shap_waterfall_plot(result):

    fig = plt.figure(figsize=(10, 6))

    shap.plots.waterfall(
        result["explanation"],
        max_display=10,
        show=False
    )

    fig = plt.gcf()
    fig.tight_layout()

    return fig


# ============================================================
# 5. LOCAL SHAP FEATURE CONTRIBUTION BAR PLOT
# ============================================================

def create_shap_bar_plot(result):

    fig = plt.figure(figsize=(10, 6))

    shap.plots.bar(
        result["explanation"],
        max_display=10,
        show=False
    )

    fig = plt.gcf()
    fig.tight_layout()

    return fig