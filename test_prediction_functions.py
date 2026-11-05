from prediction.prediction import (
    predict_diabetes,
    predict_heart_disease,
    predict_sleep_disorder
)


print("=" * 70)
print("PREDICTION FUNCTION TEST")
print("=" * 70)


# ============================================================
# DIABETES
# ============================================================

print("\n" + "=" * 70)
print("DIABETES")
print("=" * 70)

diabetes_result = predict_diabetes(
    gender="Male",
    age=60,
    hypertension=1,
    heart_disease=0,
    smoking_history="former",
    bmi=35.97,
    hba1c_level=6.6,
    blood_glucose_level=130
)

print("Input:")
print("Age                 :", 60)
print("Hypertension        :", 1)
print("Heart Disease       :", 0)
print("Smoking History     :", "former")
print("BMI                 :", 35.97)
print("HbA1c               :", 6.6)
print("Blood Glucose       :", 130)

print("\nResult:")
print("Prediction          :", diabetes_result["prediction"])
print(
    "Risk                :",
    f"{diabetes_result['probability'] * 100:.2f}%"
)


# ============================================================
# HEART DISEASE
# ============================================================

print("\n" + "=" * 70)
print("HEART DISEASE")
print("=" * 70)

heart_result = predict_heart_disease(
    age=50,
    gender="Female",
    cholesterol=230,
    blood_pressure=130,
    heart_rate=75,
    smoking="Former",
    alcohol_intake="Moderate",
    exercise_hours=3,
    family_history="Yes",
    diabetes="No",
    obesity="Yes",
    stress_level=5,
    blood_sugar=120,
    exercise_induced_angina="No",
    chest_pain_type="Atypical Angina"
)

print("Input:")
print("Age                 :", 50)
print("Gender              :", "Female")
print("Cholesterol         :", 230)
print("Blood Pressure      :", 130)
print("Heart Rate          :", 75)
print("Smoking             :", "Former")
print("Alcohol Intake      :", "Moderate")
print("Exercise Hours      :", 3)
print("Family History      :", "Yes")
print("Diabetes            :", "No")
print("Obesity             :", "Yes")
print("Stress Level        :", 5)
print("Blood Sugar         :", 120)
print("Exercise Angina     :", "No")
print("Chest Pain          :", "Atypical Angina")

print("\nResult:")
print("Prediction          :", heart_result["prediction"])
print(
    "Risk                :",
    f"{heart_result['probability'] * 100:.2f}%"
)


# ============================================================
# SLEEP DISORDER
# ============================================================

print("\n" + "=" * 70)
print("SLEEP DISORDER")
print("=" * 70)

sleep_result = predict_sleep_disorder(
    person_id=1,
    gender="Female",
    age=30,
    occupation="Software Engineer",
    sleep_duration=8,
    quality_of_sleep=8,
    physical_activity_level=80,
    stress_level=2,
    bmi_category="Normal",
    blood_pressure="120/80",
    heart_rate=65,
    daily_steps=9000
)

print("Input:")
print("Person ID           :", 1)
print("Gender              :", "Female")
print("Age                 :", 30)
print("Occupation          :", "Software Engineer")
print("Sleep Duration      :", 8)
print("Quality of Sleep    :", 8)
print("Physical Activity   :", 80)
print("Stress Level        :", 2)
print("BMI Category        :", "Normal")
print("Blood Pressure      :", "120/80")
print("Heart Rate          :", 65)
print("Daily Steps         :", 9000)

print("\nResult:")
print("Prediction          :", sleep_result["prediction"])
print(
    "Sleep Apnea Risk    :",
    f"{sleep_result['probability'] * 100:.2f}%"
)


# ============================================================
# COMPLETED
# ============================================================

print("\n" + "=" * 70)
print("ALL PREDICTION FUNCTIONS TESTED SUCCESSFULLY")
print("=" * 70)