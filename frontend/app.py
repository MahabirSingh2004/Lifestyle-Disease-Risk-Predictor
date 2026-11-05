import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import shap


from prediction.prediction import (
    predict_diabetes,
    predict_heart_disease,
    predict_sleep_disorder
)

from explainability.shap_explainer import (
    explain_diabetes,
    explain_heart_disease,
    explain_sleep_disorder
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Lifestyle Disease Risk Predictor",
    page_icon="🩺",
    layout="wide"
)


# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

if "actual_diabetes" not in st.session_state:
    st.session_state.actual_diabetes = None

if "actual_heart" not in st.session_state:
    st.session_state.actual_heart = None

if "actual_sleep" not in st.session_state:
    st.session_state.actual_sleep = None


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def save_simulator_inputs(widget_values):
    """
    Update simulator widget values using the original
    prediction inputs.

    This function is called when the user makes a new
    prediction in the disease prediction tabs.
    """

    for key, value in widget_values.items():
        st.session_state[key] = value


def get_risk_change(actual_probability, simulated_probability):
    """
    Calculate the difference between actual and simulated
    probabilities.

    Positive value = risk decreased.
    Negative value = risk increased.
    """

    return actual_probability - simulated_probability


def display_risk_change(risk_change, disease_name):

    if risk_change > 0:

        st.success(
            f"🎯 The simulated {disease_name} risk decreased "
            f"by {risk_change:.2f} percentage points."
        )

    elif risk_change < 0:

        st.warning(
            f"⚠️ The simulated {disease_name} risk increased "
            f"by {abs(risk_change):.2f} percentage points."
        )

    else:

        st.info(
            "ℹ️ There is no change in the predicted probability."
        )


def display_risk_comparison(
    actual_probability,
    simulated_probability,
    title,
    y_axis_title
):

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=[
                "Actual Risk",
                "Simulated Risk"
            ],
            y=[
                actual_probability,
                simulated_probability
            ],
            text=[
                f"{actual_probability:.2f}%",
                f"{simulated_probability:.2f}%"
            ],
            textposition="auto"
        )
    )

    chart_max = max(
        actual_probability,
        simulated_probability
    )

    fig.update_layout(
        title=title,
        yaxis_title=y_axis_title,
        xaxis_title="Risk Type",
        yaxis=dict(
            range=[
                0,
                min(100, max(5, chart_max + 5))
            ]
        ),
        height=400
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# SHAP EXPLANATION DISPLAY
# ============================================================

def display_shap_explanation(shap_result, disease_name):

    st.subheader(f"🧠 Explainable AI – {disease_name}")

    st.metric(
        "Model Probability Explained",
        f"{shap_result['prediction_probability'] * 100:.2f}%"
    )

    st.info(
        "SHAP explains how the model's features contributed "
        "to this prediction relative to the baseline. These "
        "contributions do not establish medical causation."
    )

    explanation = shap_result["explanation"]

    if len(explanation.shape) > 1:
        explanation = explanation[0]

    # --------------------------------------------------------
    # WATERFALL PLOT
    # --------------------------------------------------------

    st.markdown("### SHAP Waterfall Plot")

    try:

        fig, ax = plt.subplots(figsize=(10, 6))

        shap.plots.waterfall(
            explanation,
            max_display=10,
            show=False
        )

        st.pyplot(
            fig,
            clear_figure=True
        )

        plt.close(fig)

    except Exception as error:

        st.warning(
            f"Could not display waterfall plot: {error}"
        )

    # --------------------------------------------------------
    # FEATURE CONTRIBUTION CHART
    # --------------------------------------------------------

    st.markdown("### Top Feature Contributions")

    try:

        values = explanation.values

        if hasattr(values, "reshape"):
            values = values.reshape(-1)

        feature_names = explanation.feature_names

        if feature_names is None:
            feature_names = shap_result["feature_names"]

        feature_names = list(feature_names)

        contributions = sorted(
            zip(feature_names, values),
            key=lambda item: abs(float(item[1])),
            reverse=True
        )[:10]

        chart_names = [
            item[0] for item in contributions
        ][::-1]

        chart_values = [
            float(item[1]) for item in contributions
        ][::-1]

        contribution_fig = go.Figure(
            go.Bar(
                x=chart_values,
                y=chart_names,
                orientation="h",
                text=[
                    f"{value:+.4f}"
                    for value in chart_values
                ],
                textposition="auto"
            )
        )

        contribution_fig.update_layout(
            title="Feature Impact on Model Output",
            xaxis_title="SHAP Contribution",
            yaxis_title="Feature",
            height=450,
            showlegend=False
        )

        st.plotly_chart(
            contribution_fig,
            use_container_width=True
        )

        # ----------------------------------------------------
        # TOP FIVE FEATURE INTERPRETATION
        # ----------------------------------------------------

        st.markdown("### Top 5 Important Features")

        for feature_name, contribution in contributions[:5]:

            contribution = float(contribution)

            if contribution > 0:
                direction = "increased the model output"

            elif contribution < 0:
                direction = "decreased the model output"

            else:
                direction = "had almost no effect"

            st.write(
                f"- **{feature_name}**: "
                f"{contribution:+.4f} "
                f"({direction} relative to baseline)."
            )

    except Exception as error:

        st.error(
            f"Could not display feature contributions: {error}"
        )


# ============================================================
# MAIN TITLE
# ============================================================

st.title("🩺 Lifestyle Risk Disease Predictor")

st.write(
    "Predict lifestyle-related health risks using "
    "machine learning models."
)


# ============================================================
# HOME / DASHBOARD
# ============================================================

st.divider()

st.header("🏠 Welcome to the Lifestyle Risk Disease Predictor")

st.write(
    "This application uses machine learning models to analyze "
    "health and lifestyle factors and predict the risk of selected "
    "lifestyle-related diseases."
)

st.subheader("📌 Available Prediction Modules")

col1, col2, col3 = st.columns(3)

with col1:

    st.info(
        "🩸 **Diabetes**\n\n"
        "Predict diabetes risk using age, BMI, glucose level, "
        "HbA1c level and other health factors."
    )

with col2:

    st.info(
        "❤️ **Heart Disease**\n\n"
        "Analyze cardiovascular health factors such as cholesterol, "
        "blood pressure, heart rate and lifestyle habits."
    )

with col3:

    st.info(
        "😴 **Sleep Disorder**\n\n"
        "Predict sleep-related disorders using sleep duration, "
        "quality, stress, activity and other lifestyle factors."
    )

st.subheader("🔄 How the System Works")

st.write(
    "Enter your health and lifestyle information → "
    "Machine Learning Model → Risk Prediction → "
    "View the Prediction Result → SHAP Explanation → "
    "Habit Simulation"
)

st.divider()


# ============================================================
# DISEASE TABS
# ============================================================

tab1, tab2, tab3 = st.tabs(
    [
        "🩸 Diabetes",
        "❤️ Heart Disease",
        "😴 Sleep Disorder"
    ]
)


# ============================================================
# DIABETES PREDICTION
# ============================================================

with tab1:

    st.header("🩸 Diabetes Risk Prediction")

    col1, col2 = st.columns(2)

    with col1:

        gender = st.selectbox(
            "Gender",
            ["Female", "Male", "Other"],
            key="diabetes_gender"
        )

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=40,
            key="diabetes_age"
        )

        hypertension = st.selectbox(
            "Hypertension",
            ["No", "Yes"],
            key="diabetes_hypertension"
        )

        heart_disease = st.selectbox(
            "Heart Disease",
            ["No", "Yes"],
            key="diabetes_heart_disease"
        )

    with col2:

        smoking_history = st.selectbox(
            "Smoking History",
            [
                "never",
                "No Info",
                "current",
                "former",
                "ever",
                "not current"
            ],
            key="diabetes_smoking"
        )

        bmi = st.number_input(
            "BMI",
            min_value=10.0,
            max_value=70.0,
            value=24.5,
            step=0.1,
            key="diabetes_bmi"
        )

        hba1c_level = st.number_input(
            "HbA1c Level",
            min_value=3.0,
            max_value=15.0,
            value=5.5,
            step=0.1,
            key="diabetes_hba1c"
        )

        blood_glucose_level = st.number_input(
            "Blood Glucose Level",
            min_value=50,
            max_value=500,
            value=100,
            key="diabetes_glucose"
        )

    # --------------------------------------------------------
    # DIABETES PREDICTION
    # --------------------------------------------------------

    if st.button(
        "🔍 Predict Diabetes Risk",
        key="predict_diabetes"
    ):

        result = predict_diabetes(
            gender=gender,
            age=age,
            hypertension=1 if hypertension == "Yes" else 0,
            heart_disease=1 if heart_disease == "Yes" else 0,
            smoking_history=smoking_history,
            bmi=bmi,
            hba1c_level=hba1c_level,
            blood_glucose_level=blood_glucose_level
        )

        probability = result["probability"] * 100

        # Save original prediction independently.
        st.session_state.actual_diabetes = {
            "inputs": {
                "gender": gender,
                "age": age,
                "hypertension": hypertension,
                "heart_disease": heart_disease,
                "smoking_history": smoking_history,
                "bmi": bmi,
                "hba1c_level": hba1c_level,
                "blood_glucose_level": blood_glucose_level
            },
            "prediction": result["prediction"],
            "probability": probability
        }

        # Initialize simulator with original inputs.
        save_simulator_inputs(
            {
                "sim_diabetes_gender": gender,
                "sim_diabetes_age": age,
                "sim_diabetes_hypertension": hypertension,
                "sim_diabetes_heart": heart_disease,
                "sim_diabetes_smoking": smoking_history,
                "sim_diabetes_bmi": bmi,
                "sim_diabetes_hba1c": hba1c_level,
                "sim_diabetes_glucose": blood_glucose_level
            }
        )

        st.session_state.prediction_history.append(
            {
                "disease": "Diabetes",
                "prediction": (
                    "Higher Risk"
                    if result["prediction"] == 1
                    else "Lower Risk"
                ),
                "probability": probability
            }
        )

        st.subheader("Prediction Result")

        if result["prediction"] == 1:

            st.error("⚠️ Higher Diabetes Risk")

        else:

            st.success("✅ Lower Diabetes Risk")

        st.metric(
            "Diabetes Probability",
            f"{probability:.2f}%"
        )

        st.success(
            "Your original diabetes prediction has been saved "
            "and will be used as the Actual Risk in the Habit Simulator."
        )

    # --------------------------------------------------------
    # DIABETES SHAP EXPLANATION
    # --------------------------------------------------------

    st.divider()

    if st.button(
        "🧠 Explain Current Inputs with SHAP",
        key="explain_diabetes"
    ):

        with st.spinner("Generating diabetes explanation..."):

            try:

                shap_result = explain_diabetes(
                    gender=gender,
                    age=age,
                    hypertension=(
                        1 if hypertension == "Yes" else 0
                    ),
                    heart_disease=(
                        1 if heart_disease == "Yes" else 0
                    ),
                    smoking_history=smoking_history,
                    bmi=bmi,
                    hba1c_level=hba1c_level,
                    blood_glucose_level=blood_glucose_level
                )

                display_shap_explanation(
                    shap_result,
                    "Diabetes"
                )

            except Exception as error:

                st.error(
                    f"Unable to generate diabetes explanation: {error}"
                )


# ============================================================
# HEART DISEASE PREDICTION
# ============================================================

with tab2:

    st.header("❤️ Heart Disease Risk Prediction")

    col1, col2 = st.columns(2)

    with col1:

        heart_age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=50,
            key="heart_age"
        )

        heart_gender = st.selectbox(
            "Gender",
            ["Female", "Male"],
            key="heart_gender"
        )

        cholesterol = st.number_input(
            "Cholesterol",
            min_value=50,
            max_value=500,
            value=200,
            key="cholesterol"
        )

        blood_pressure = st.number_input(
            "Blood Pressure",
            min_value=50,
            max_value=250,
            value=120,
            key="heart_bp"
        )

        heart_rate = st.number_input(
            "Heart Rate",
            min_value=30,
            max_value=200,
            value=70,
            key="heart_rate"
        )

        smoking = st.selectbox(
            "Smoking",
            ["Current", "Never", "Former"],
            key="heart_smoking"
        )

        alcohol_intake = st.selectbox(
            "Alcohol Intake",
            ["Heavy", "Moderate"],
            key="alcohol"
        )

        exercise_hours = st.number_input(
            "Exercise Hours",
            min_value=0.0,
            max_value=20.0,
            value=5.0,
            step=0.5,
            key="exercise_hours"
        )

    with col2:

        family_history = st.selectbox(
            "Family History",
            ["No", "Yes"],
            key="family_history"
        )

        diabetes = st.selectbox(
            "Diabetes",
            ["No", "Yes"],
            key="heart_diabetes"
        )

        obesity = st.selectbox(
            "Obesity",
            ["Yes", "No"],
            key="obesity"
        )

        stress_level = st.number_input(
            "Stress Level",
            min_value=0,
            max_value=10,
            value=4,
            key="stress_level"
        )

        blood_sugar = st.number_input(
            "Blood Sugar",
            min_value=50,
            max_value=500,
            value=100,
            key="blood_sugar"
        )

        exercise_induced_angina = st.selectbox(
            "Exercise Induced Angina",
            ["Yes", "No"],
            key="angina"
        )

        chest_pain_type = st.selectbox(
            "Chest Pain Type",
            [
                "Atypical Angina",
                "Typical Angina",
                "Non-anginal Pain",
                "Asymptomatic"
            ],
            key="chest_pain"
        )

    # --------------------------------------------------------
    # HEART DISEASE PREDICTION
    # --------------------------------------------------------

    if st.button(
        "❤️ Predict Heart Disease Risk",
        key="predict_heart"
    ):

        result = predict_heart_disease(
            age=heart_age,
            gender=heart_gender,
            cholesterol=cholesterol,
            blood_pressure=blood_pressure,
            heart_rate=heart_rate,
            smoking=smoking,
            alcohol_intake=alcohol_intake,
            exercise_hours=exercise_hours,
            family_history=family_history,
            diabetes=diabetes,
            obesity=obesity,
            stress_level=stress_level,
            blood_sugar=blood_sugar,
            exercise_induced_angina=exercise_induced_angina,
            chest_pain_type=chest_pain_type
        )

        probability = result["probability"] * 100

        # Save original prediction independently.
        st.session_state.actual_heart = {
            "inputs": {
                "age": heart_age,
                "gender": heart_gender,
                "cholesterol": cholesterol,
                "blood_pressure": blood_pressure,
                "heart_rate": heart_rate,
                "smoking": smoking,
                "alcohol_intake": alcohol_intake,
                "exercise_hours": exercise_hours,
                "family_history": family_history,
                "diabetes": diabetes,
                "obesity": obesity,
                "stress_level": stress_level,
                "blood_sugar": blood_sugar,
                "exercise_induced_angina": exercise_induced_angina,
                "chest_pain_type": chest_pain_type
            },
            "prediction": result["prediction"],
            "probability": probability
        }

        # Initialize simulator with original inputs.
        save_simulator_inputs(
            {
                "sim_heart_age": heart_age,
                "sim_heart_gender": heart_gender,
                "sim_cholesterol": cholesterol,
                "sim_blood_pressure": blood_pressure,
                "sim_heart_rate": heart_rate,
                "sim_heart_smoking": smoking,
                "sim_alcohol": alcohol_intake,
                "sim_exercise_hours": exercise_hours,
                "sim_family_history": family_history,
                "sim_heart_diabetes": diabetes,
                "sim_obesity": obesity,
                "sim_stress": stress_level,
                "sim_blood_sugar": blood_sugar,
                "sim_angina": exercise_induced_angina,
                "sim_chest_pain": chest_pain_type
            }
        )

        st.session_state.prediction_history.append(
            {
                "disease": "Heart Disease",
                "prediction": (
                    "Higher Risk"
                    if result["prediction"] == 1
                    else "Lower Risk"
                ),
                "probability": probability
            }
        )

        st.subheader("Prediction Result")

        if result["prediction"] == 1:

            st.error("⚠️ Higher Heart Disease Risk")

        else:

            st.success("✅ Lower Heart Disease Risk")

        st.metric(
            "Heart Disease Probability",
            f"{probability:.2f}%"
        )

        st.success(
            "Your original heart disease prediction has been saved "
            "and will be used as the Actual Risk in the Habit Simulator."
        )

    # --------------------------------------------------------
    # HEART DISEASE SHAP EXPLANATION
    # --------------------------------------------------------

    st.divider()

    if st.button(
        "🧠 Explain Current Inputs with SHAP",
        key="explain_heart"
    ):

        with st.spinner("Generating heart disease explanation..."):

            try:

                shap_result = explain_heart_disease(
                    age=heart_age,
                    gender=heart_gender,
                    cholesterol=cholesterol,
                    blood_pressure=blood_pressure,
                    heart_rate=heart_rate,
                    smoking=smoking,
                    alcohol_intake=alcohol_intake,
                    exercise_hours=exercise_hours,
                    family_history=family_history,
                    diabetes=diabetes,
                    obesity=obesity,
                    stress_level=stress_level,
                    blood_sugar=blood_sugar,
                    exercise_induced_angina=exercise_induced_angina,
                    chest_pain_type=chest_pain_type
                )

                display_shap_explanation(
                    shap_result,
                    "Heart Disease"
                )

            except Exception as error:

                st.error(
                    f"Unable to generate heart disease explanation: {error}"
                )


# ============================================================
# SLEEP DISORDER PREDICTION
# ============================================================

with tab3:

    st.header("😴 Sleep Disorder Prediction")

    col1, col2 = st.columns(2)

    with col1:

        person_id = st.number_input(
            "Person ID",
            min_value=1,
            max_value=1000,
            value=100,
            key="person_id"
        )

        sleep_gender = st.selectbox(
            "Gender",
            ["Male", "Female"],
            key="sleep_gender"
        )

        sleep_age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=30,
            key="sleep_age"
        )

        occupation = st.selectbox(
            "Occupation",
            [
                "Software Engineer",
                "Doctor",
                "Sales Representative",
                "Teacher",
                "Nurse",
                "Engineer",
                "Accountant",
                "Scientist",
                "Lawyer",
                "Salesperson",
                "Manager"
            ],
            key="occupation"
        )

        sleep_duration = st.number_input(
            "Sleep Duration (hours)",
            min_value=1.0,
            max_value=15.0,
            value=6.5,
            step=0.1,
            key="sleep_duration"
        )

        quality_of_sleep = st.number_input(
            "Quality of Sleep",
            min_value=1,
            max_value=10,
            value=6,
            key="quality_sleep"
        )

    with col2:

        physical_activity_level = st.number_input(
            "Physical Activity Level",
            min_value=0,
            max_value=150,
            value=45,
            key="physical_activity"
        )

        sleep_stress_level = st.number_input(
            "Stress Level",
            min_value=1,
            max_value=10,
            value=6,
            key="sleep_stress"
        )

        bmi_category = st.selectbox(
            "BMI Category",
            [
                "Overweight",
                "Normal",
                "Obese",
                "Normal Weight"
            ],
            key="bmi_category"
        )

        sleep_blood_pressure = st.selectbox(
            "Blood Pressure",
            [
                "115/75",
                "115/78",
                "117/76",
                "118/75",
                "118/76",
                "119/77",
                "120/80",
                "121/79",
                "122/80",
                "125/80",
                "125/82",
                "126/83",
                "128/84",
                "128/85",
                "129/84",
                "130/85",
                "130/86",
                "131/86",
                "132/87",
                "135/88",
                "135/90",
                "139/91",
                "140/90",
                "140/95",
                "142/92"
            ],
            key="sleep_bp"
        )

        sleep_heart_rate = st.number_input(
            "Heart Rate",
            min_value=30,
            max_value=200,
            value=75,
            key="sleep_heart_rate"
        )

        daily_steps = st.number_input(
            "Daily Steps",
            min_value=0,
            max_value=30000,
            value=5000,
            key="daily_steps"
        )

    # --------------------------------------------------------
    # SLEEP DISORDER PREDICTION
    # --------------------------------------------------------

    if st.button(
        "😴 Predict Sleep Disorder",
        key="predict_sleep"
    ):

        result = predict_sleep_disorder(
            person_id=person_id,
            gender=sleep_gender,
            age=sleep_age,
            occupation=occupation,
            sleep_duration=sleep_duration,
            quality_of_sleep=quality_of_sleep,
            physical_activity_level=physical_activity_level,
            stress_level=sleep_stress_level,
            bmi_category=bmi_category,
            blood_pressure=sleep_blood_pressure,
            heart_rate=sleep_heart_rate,
            daily_steps=daily_steps
        )

        prediction = result["prediction"]

        sleep_apnea_probability = (
            result["probability"] * 100
        )

        # Save original prediction independently.
        st.session_state.actual_sleep = {
            "inputs": {
                "person_id": person_id,
                "gender": sleep_gender,
                "age": sleep_age,
                "occupation": occupation,
                "sleep_duration": sleep_duration,
                "quality_of_sleep": quality_of_sleep,
                "physical_activity_level": physical_activity_level,
                "stress_level": sleep_stress_level,
                "bmi_category": bmi_category,
                "blood_pressure": sleep_blood_pressure,
                "heart_rate": sleep_heart_rate,
                "daily_steps": daily_steps
            },
            "prediction": prediction,
            "probability": sleep_apnea_probability
        }

        # Initialize simulator with original inputs.
        save_simulator_inputs(
            {
                "sim_sleep_gender": sleep_gender,
                "sim_sleep_age": sleep_age,
                "sim_occupation": occupation,
                "sim_sleep_duration": sleep_duration,
                "sim_quality": quality_of_sleep,
                "sim_activity": physical_activity_level,
                "sim_sleep_stress": sleep_stress_level,
                "sim_bmi_category": bmi_category,
                "sim_sleep_bp": sleep_blood_pressure,
                "sim_sleep_hr": sleep_heart_rate,
                "sim_steps": daily_steps
            }
        )

        st.session_state.prediction_history.append(
            {
                "disease": "Sleep Disorder",
                "prediction": prediction,
                "probability": sleep_apnea_probability
            }
        )

        st.subheader("Prediction Result")

        st.info(
            f"🛌 Predicted Sleep Disorder: **{prediction}**"
        )

        st.metric(
            "Sleep Apnea Risk",
            f"{sleep_apnea_probability:.2f}%"
        )

        st.success(
            "Your original sleep disorder prediction has been saved "
            "and will be used as the Actual Result in the Habit Simulator."
        )

    # --------------------------------------------------------
    # SLEEP DISORDER SHAP EXPLANATION
    # --------------------------------------------------------

    st.divider()

    if st.button(
        "🧠 Explain Current Inputs with SHAP",
        key="explain_sleep"
    ):

        with st.spinner("Generating sleep disorder explanation..."):

            try:

                shap_result = explain_sleep_disorder(
                    person_id=person_id,
                    gender=sleep_gender,
                    age=sleep_age,
                    occupation=occupation,
                    sleep_duration=sleep_duration,
                    quality_of_sleep=quality_of_sleep,
                    physical_activity_level=physical_activity_level,
                    stress_level=sleep_stress_level,
                    bmi_category=bmi_category,
                    blood_pressure=sleep_blood_pressure,
                    heart_rate=sleep_heart_rate,
                    daily_steps=daily_steps
                )

                display_shap_explanation(
                    shap_result,
                    "Sleep Disorder"
                )

            except Exception as error:

                st.error(
                    f"Unable to generate sleep disorder explanation: {error}"
                )


# ============================================================
# PREDICTION HISTORY
# ============================================================

st.divider()

st.header("📋 Prediction History")

if st.session_state.prediction_history:

    for item in st.session_state.prediction_history:

        st.write(
            f"**{item['disease']}** | "
            f"Prediction: **{item['prediction']}** | "
            f"Probability/Risk: "
            f"**{item['probability']:.2f}%**"
        )

    if st.button("🗑️ Clear Prediction History"):

        st.session_state.prediction_history = []

        st.rerun()

else:

    st.info("No predictions have been made yet.")


# ============================================================
# HABIT SIMULATOR
# ============================================================

st.divider()

st.header("🔄 Habit Simulator")

st.write(
    "Modify your health and lifestyle factors to understand "
    "how the trained machine learning model responds to changes."
)

st.info(
    "**Actual Risk** is taken from your original prediction. "
    "**Simulated Risk** is calculated using your modified inputs. "
    "The original prediction remains unchanged when you modify "
    "the simulator controls."
)


# ============================================================
# SIMULATION DISEASE SELECTION
# ============================================================

simulation_disease = st.selectbox(
    "Select Simulation Category",
    [
        "🩸 Diabetes",
        "❤️ Heart Disease",
        "😴 Sleep Disorder"
    ],
    key="simulation_disease"
)


# ============================================================
# DIABETES HABIT SIMULATOR
# ============================================================

if simulation_disease == "🩸 Diabetes":

    st.subheader("🩸 Diabetes Habit Simulation")

    actual_diabetes = st.session_state.actual_diabetes

    if actual_diabetes is None:

        st.warning(
            "Please complete Diabetes Risk Prediction first. "
            "The simulator needs your original diabetes prediction "
            "to calculate Actual Risk."
        )

    else:

        st.write(
            "The simulator starts with the same inputs used in your "
            "original Diabetes Risk Prediction. You can now change "
            "the values and compare the results."
        )

        # ----------------------------------------------------
        # SHOW ACTUAL PREDICTION
        # ----------------------------------------------------

        st.markdown("### Original Diabetes Prediction")

        actual_col1, actual_col2 = st.columns(2)

        with actual_col1:

            st.metric(
                "Actual Diabetes Risk",
                f"{actual_diabetes['probability']:.2f}%"
            )

        with actual_col2:

            actual_status = (
                "Higher Risk"
                if actual_diabetes["prediction"] == 1
                else "Lower Risk"
            )

            st.metric(
                "Actual Prediction",
                actual_status
            )

        st.divider()

        st.markdown("### Modify Lifestyle and Health Factors")

        col1, col2 = st.columns(2)

        with col1:

            sim_diabetes_gender = st.selectbox(
                "Gender",
                ["Female", "Male", "Other"],
                key="sim_diabetes_gender",
                index=(
                    ["Female", "Male", "Other"].index(
                        st.session_state.get(
                            "sim_diabetes_gender",
                            actual_diabetes["inputs"]["gender"]
                        )
                    )
                )
            )

            sim_diabetes_age = st.slider(
                "Age",
                min_value=1,
                max_value=120,
                key="sim_diabetes_age",
                value=int(
                    st.session_state.get(
                        "sim_diabetes_age",
                        actual_diabetes["inputs"]["age"]
                    )
                )
            )

            sim_diabetes_hypertension = st.selectbox(
                "Hypertension",
                ["No", "Yes"],
                key="sim_diabetes_hypertension"
            )

            sim_diabetes_heart = st.selectbox(
                "Heart Disease",
                ["No", "Yes"],
                key="sim_diabetes_heart"
            )

        with col2:

            sim_diabetes_smoking = st.selectbox(
                "Smoking History",
                [
                    "never",
                    "No Info",
                    "current",
                    "former",
                    "ever",
                    "not current"
                ],
                key="sim_diabetes_smoking"
            )

            sim_diabetes_bmi = st.slider(
                "BMI",
                min_value=10.0,
                max_value=70.0,
                key="sim_diabetes_bmi",
                value=float(
                    st.session_state.get(
                        "sim_diabetes_bmi",
                        actual_diabetes["inputs"]["bmi"]
                    )
                ),
                step=0.1
            )

            sim_diabetes_hba1c = st.slider(
                "HbA1c Level",
                min_value=3.0,
                max_value=15.0,
                key="sim_diabetes_hba1c",
                value=float(
                    st.session_state.get(
                        "sim_diabetes_hba1c",
                        actual_diabetes["inputs"]["hba1c_level"]
                    )
                ),
                step=0.1
            )

            sim_diabetes_glucose = st.slider(
                "Blood Glucose Level",
                min_value=50,
                max_value=500,
                key="sim_diabetes_glucose",
                value=int(
                    st.session_state.get(
                        "sim_diabetes_glucose",
                        actual_diabetes["inputs"]["blood_glucose_level"]
                    )
                ),
                step=5
            )

        # ----------------------------------------------------
        # DIABETES SIMULATION
        # ----------------------------------------------------

        if st.button(
            "🔍 Simulate Diabetes Risk",
            key="simulate_diabetes"
        ):

            simulated_diabetes = predict_diabetes(
                gender=sim_diabetes_gender,
                age=sim_diabetes_age,
                hypertension=(
                    1 if sim_diabetes_hypertension == "Yes" else 0
                ),
                heart_disease=(
                    1 if sim_diabetes_heart == "Yes" else 0
                ),
                smoking_history=sim_diabetes_smoking,
                bmi=sim_diabetes_bmi,
                hba1c_level=sim_diabetes_hba1c,
                blood_glucose_level=sim_diabetes_glucose
            )

            actual_probability = actual_diabetes["probability"]

            simulated_probability = (
                simulated_diabetes["probability"] * 100
            )

            risk_change = get_risk_change(
                actual_probability,
                simulated_probability
            )

            st.divider()

            st.subheader("📊 Diabetes Simulation Result")

            result_col1, result_col2, result_col3 = st.columns(3)

            with result_col1:

                st.metric(
                    "Actual Risk",
                    f"{actual_probability:.2f}%"
                )

            with result_col2:

                st.metric(
                    "Simulated Risk",
                    f"{simulated_probability:.2f}%"
                )

            with result_col3:

                st.metric(
                    "Risk Change",
                    f"{risk_change:+.2f} pp"
                )

            if simulated_diabetes["prediction"] == 1:

                st.error("⚠️ Higher Diabetes Risk")

            else:

                st.success("✅ Lower Diabetes Risk")

            display_risk_change(
                risk_change,
                "Diabetes"
            )

            display_risk_comparison(
                actual_probability,
                simulated_probability,
                "Actual vs Simulated Diabetes Risk",
                "Diabetes Probability (%)"
            )


# ============================================================
# HEART DISEASE HABIT SIMULATOR
# ============================================================

elif simulation_disease == "❤️ Heart Disease":

    st.subheader("❤️ Heart Disease Habit Simulation")

    actual_heart = st.session_state.actual_heart

    if actual_heart is None:

        st.warning(
            "Please complete Heart Disease Risk Prediction first. "
            "The simulator needs your original heart disease prediction "
            "to calculate Actual Risk."
        )

    else:

        st.write(
            "The simulator starts with the same inputs used in your "
            "original Heart Disease Risk Prediction."
        )

        # ----------------------------------------------------
        # SHOW ACTUAL PREDICTION
        # ----------------------------------------------------

        st.markdown("### Original Heart Disease Prediction")

        actual_col1, actual_col2 = st.columns(2)

        with actual_col1:

            st.metric(
                "Actual Heart Disease Risk",
                f"{actual_heart['probability']:.2f}%"
            )

        with actual_col2:

            actual_status = (
                "Higher Risk"
                if actual_heart["prediction"] == 1
                else "Lower Risk"
            )

            st.metric(
                "Actual Prediction",
                actual_status
            )

        st.divider()

        st.markdown("### Modify Lifestyle and Health Factors")

        col1, col2 = st.columns(2)

        with col1:

            sim_heart_age = st.slider(
                "Age",
                min_value=1,
                max_value=120,
                key="sim_heart_age",
                value=int(
                    st.session_state.get(
                        "sim_heart_age",
                        actual_heart["inputs"]["age"]
                    )
                )
            )

            sim_heart_gender = st.selectbox(
                "Gender",
                ["Female", "Male"],
                key="sim_heart_gender"
            )

            sim_cholesterol = st.slider(
                "Cholesterol",
                min_value=50,
                max_value=500,
                key="sim_cholesterol",
                value=int(
                    st.session_state.get(
                        "sim_cholesterol",
                        actual_heart["inputs"]["cholesterol"]
                    )
                )
            )

            sim_blood_pressure = st.slider(
                "Blood Pressure",
                min_value=50,
                max_value=250,
                key="sim_blood_pressure",
                value=int(
                    st.session_state.get(
                        "sim_blood_pressure",
                        actual_heart["inputs"]["blood_pressure"]
                    )
                )
            )

            sim_heart_rate = st.slider(
                "Heart Rate",
                min_value=30,
                max_value=200,
                key="sim_heart_rate",
                value=int(
                    st.session_state.get(
                        "sim_heart_rate",
                        actual_heart["inputs"]["heart_rate"]
                    )
                )
            )

            sim_exercise_hours = st.slider(
                "Exercise Hours",
                min_value=0.0,
                max_value=20.0,
                key="sim_exercise_hours",
                value=float(
                    st.session_state.get(
                        "sim_exercise_hours",
                        actual_heart["inputs"]["exercise_hours"]
                    )
                ),
                step=0.5
            )

            sim_stress = st.slider(
                "Stress Level",
                min_value=0,
                max_value=10,
                key="sim_stress",
                value=int(
                    st.session_state.get(
                        "sim_stress",
                        actual_heart["inputs"]["stress_level"]
                    )
                )
            )

        with col2:

            sim_heart_smoking = st.selectbox(
                "Smoking",
                ["Current", "Never", "Former"],
                key="sim_heart_smoking"
            )

            sim_alcohol = st.selectbox(
                "Alcohol Intake",
                ["Heavy", "Moderate"],
                key="sim_alcohol"
            )

            sim_family_history = st.selectbox(
                "Family History",
                ["No", "Yes"],
                key="sim_family_history"
            )

            sim_heart_diabetes = st.selectbox(
                "Diabetes",
                ["No", "Yes"],
                key="sim_heart_diabetes"
            )

            sim_obesity = st.selectbox(
                "Obesity",
                ["Yes", "No"],
                key="sim_obesity"
            )

            sim_blood_sugar = st.slider(
                "Blood Sugar",
                min_value=50,
                max_value=500,
                key="sim_blood_sugar",
                value=int(
                    st.session_state.get(
                        "sim_blood_sugar",
                        actual_heart["inputs"]["blood_sugar"]
                    )
                )
            )

            sim_angina = st.selectbox(
                "Exercise Induced Angina",
                ["Yes", "No"],
                key="sim_angina"
            )

            sim_chest_pain = st.selectbox(
                "Chest Pain Type",
                [
                    "Atypical Angina",
                    "Typical Angina",
                    "Non-anginal Pain",
                    "Asymptomatic"
                ],
                key="sim_chest_pain"
            )

        # ----------------------------------------------------
        # HEART DISEASE SIMULATION
        # ----------------------------------------------------

        if st.button(
            "🔍 Simulate Heart Disease Risk",
            key="simulate_heart"
        ):

            simulated_heart = predict_heart_disease(
                age=sim_heart_age,
                gender=sim_heart_gender,
                cholesterol=sim_cholesterol,
                blood_pressure=sim_blood_pressure,
                heart_rate=sim_heart_rate,
                smoking=sim_heart_smoking,
                alcohol_intake=sim_alcohol,
                exercise_hours=sim_exercise_hours,
                family_history=sim_family_history,
                diabetes=sim_heart_diabetes,
                obesity=sim_obesity,
                stress_level=sim_stress,
                blood_sugar=sim_blood_sugar,
                exercise_induced_angina=sim_angina,
                chest_pain_type=sim_chest_pain
            )

            actual_probability = actual_heart["probability"]

            simulated_probability = (
                simulated_heart["probability"] * 100
            )

            risk_change = get_risk_change(
                actual_probability,
                simulated_probability
            )

            st.divider()

            st.subheader("📊 Heart Disease Simulation Result")

            result_col1, result_col2, result_col3 = st.columns(3)

            with result_col1:

                st.metric(
                    "Actual Risk",
                    f"{actual_probability:.2f}%"
                )

            with result_col2:

                st.metric(
                    "Simulated Risk",
                    f"{simulated_probability:.2f}%"
                )

            with result_col3:

                st.metric(
                    "Risk Change",
                    f"{risk_change:+.2f} pp"
                )

            if simulated_heart["prediction"] == 1:

                st.error("⚠️ Higher Heart Disease Risk")

            else:

                st.success("✅ Lower Heart Disease Risk")

            display_risk_change(
                risk_change,
                "Heart Disease"
            )

            display_risk_comparison(
                actual_probability,
                simulated_probability,
                "Actual vs Simulated Heart Disease Risk",
                "Heart Disease Probability (%)"
            )


# ============================================================
# SLEEP DISORDER HABIT SIMULATOR
# ============================================================

else:

    st.subheader("😴 Sleep Disorder Habit Simulation")

    actual_sleep = st.session_state.actual_sleep

    if actual_sleep is None:

        st.warning(
            "Please complete Sleep Disorder Prediction first. "
            "The simulator needs your original sleep disorder prediction "
            "to calculate the Actual Result."
        )

    else:

        st.write(
            "The simulator starts with the same inputs used in your "
            "original Sleep Disorder Prediction."
        )

        # ----------------------------------------------------
        # SHOW ACTUAL PREDICTION
        # ----------------------------------------------------

        st.markdown("### Original Sleep Disorder Prediction")

        actual_col1, actual_col2 = st.columns(2)

        with actual_col1:

            st.metric(
                "Actual Sleep Disorder",
                str(actual_sleep["prediction"])
            )

        with actual_col2:

            st.metric(
                "Actual Sleep Apnea Risk",
                f"{actual_sleep['probability']:.2f}%"
            )

        st.divider()

        st.markdown("### Modify Lifestyle and Health Factors")

        col1, col2 = st.columns(2)

        with col1:

            sim_sleep_gender = st.selectbox(
                "Gender",
                ["Male", "Female"],
                key="sim_sleep_gender"
            )

            sim_sleep_age = st.slider(
                "Age",
                min_value=1,
                max_value=120,
                key="sim_sleep_age",
                value=int(
                    st.session_state.get(
                        "sim_sleep_age",
                        actual_sleep["inputs"]["age"]
                    )
                )
            )

            sim_occupation = st.selectbox(
                "Occupation",
                [
                    "Software Engineer",
                    "Doctor",
                    "Sales Representative",
                    "Teacher",
                    "Nurse",
                    "Engineer",
                    "Accountant",
                    "Scientist",
                    "Lawyer",
                    "Salesperson",
                    "Manager"
                ],
                key="sim_occupation"
            )

            sim_sleep_duration = st.slider(
                "Sleep Duration (hours)",
                min_value=1.0,
                max_value=15.0,
                key="sim_sleep_duration",
                value=float(
                    st.session_state.get(
                        "sim_sleep_duration",
                        actual_sleep["inputs"]["sleep_duration"]
                    )
                ),
                step=0.1
            )

            sim_quality = st.slider(
                "Quality of Sleep",
                min_value=1,
                max_value=10,
                key="sim_quality",
                value=int(
                    st.session_state.get(
                        "sim_quality",
                        actual_sleep["inputs"]["quality_of_sleep"]
                    )
                )
            )

        with col2:

            sim_activity = st.slider(
                "Physical Activity Level",
                min_value=0,
                max_value=150,
                key="sim_activity",
                value=int(
                    st.session_state.get(
                        "sim_activity",
                        actual_sleep["inputs"]["physical_activity_level"]
                    )
                )
            )

            sim_sleep_stress = st.slider(
                "Stress Level",
                min_value=1,
                max_value=10,
                key="sim_sleep_stress",
                value=int(
                    st.session_state.get(
                        "sim_sleep_stress",
                        actual_sleep["inputs"]["stress_level"]
                    )
                )
            )

            sim_bmi_category = st.selectbox(
                "BMI Category",
                [
                    "Overweight",
                    "Normal",
                    "Obese",
                    "Normal Weight"
                ],
                key="sim_bmi_category"
            )

            sim_sleep_bp = st.selectbox(
                "Blood Pressure",
                [
                    "115/75",
                    "115/78",
                    "117/76",
                    "118/75",
                    "118/76",
                    "119/77",
                    "120/80",
                    "121/79",
                    "122/80",
                    "125/80",
                    "125/82",
                    "126/83",
                    "128/84",
                    "128/85",
                    "129/84",
                    "130/85",
                    "130/86",
                    "131/86",
                    "132/87",
                    "135/88",
                    "135/90",
                    "139/91",
                    "140/90",
                    "140/95",
                    "142/92"
                ],
                key="sim_sleep_bp"
            )

            sim_sleep_hr = st.slider(
                "Heart Rate",
                min_value=30,
                max_value=200,
                key="sim_sleep_hr",
                value=int(
                    st.session_state.get(
                        "sim_sleep_hr",
                        actual_sleep["inputs"]["heart_rate"]
                    )
                )
            )

            sim_steps = st.slider(
                "Daily Steps",
                min_value=0,
                max_value=30000,
                key="sim_steps",
                value=int(
                    st.session_state.get(
                        "sim_steps",
                        actual_sleep["inputs"]["daily_steps"]
                    )
                ),
                step=500
            )

        # ----------------------------------------------------
        # SLEEP DISORDER SIMULATION
        # ----------------------------------------------------

        if st.button(
            "🔍 Simulate Sleep Disorder",
            key="simulate_sleep"
        ):

            simulated_sleep = predict_sleep_disorder(
                person_id=actual_sleep["inputs"]["person_id"],
                gender=sim_sleep_gender,
                age=sim_sleep_age,
                occupation=sim_occupation,
                sleep_duration=sim_sleep_duration,
                quality_of_sleep=sim_quality,
                physical_activity_level=sim_activity,
                stress_level=sim_sleep_stress,
                bmi_category=sim_bmi_category,
                blood_pressure=sim_sleep_bp,
                heart_rate=sim_sleep_hr,
                daily_steps=sim_steps
            )

            actual_probability = actual_sleep["probability"]

            simulated_probability = (
                simulated_sleep["probability"] * 100
            )

            risk_change = get_risk_change(
                actual_probability,
                simulated_probability
            )

            st.divider()

            st.subheader("📊 Sleep Disorder Simulation Result")

            result_col1, result_col2, result_col3 = st.columns(3)

            with result_col1:

                st.metric(
                    "Actual Prediction",
                    str(actual_sleep["prediction"])
                )

                st.caption(
                    f"Actual Sleep Apnea Risk: "
                    f"{actual_probability:.2f}%"
                )

            with result_col2:

                st.metric(
                    "Simulated Prediction",
                    str(simulated_sleep["prediction"])
                )

                st.caption(
                    f"Simulated Sleep Apnea Risk: "
                    f"{simulated_probability:.2f}%"
                )

            with result_col3:

                st.metric(
                    "Risk Change",
                    f"{risk_change:+.2f} pp"
                )

            st.info(
                f"🛌 Simulated Sleep Disorder: "
                f"**{simulated_sleep['prediction']}**"
            )

            st.metric(
                "Simulated Sleep Apnea Risk",
                f"{simulated_probability:.2f}%"
            )

            display_risk_change(
                risk_change,
                "Sleep Apnea"
            )

            display_risk_comparison(
                actual_probability,
                simulated_probability,
                "Actual vs Simulated Sleep Apnea Risk",
                "Sleep Apnea Probability (%)"
            )


# ============================================================
# FINAL INFORMATION
# ============================================================

st.divider()

st.info(
    "💡 The Habit Simulator uses the same trained machine learning "
    "models and prediction functions as the original prediction "
    "modules. The Actual Risk is stored separately, while the "
    "Simulated Risk is calculated using the modified inputs."
)

st.caption(
    "Risk Change is calculated in percentage points by subtracting "
    "the simulated probability from the actual probability. "
    "A positive value indicates a lower simulated probability, "
    "while a negative value indicates a higher simulated probability."
)

st.caption(
    "Disclaimer: This application is an academic machine-learning "
    "project and is not a substitute for professional medical "
    "advice, diagnosis or treatment. Simulated results are model "
    "outputs and do not guarantee real-world health outcomes."
)