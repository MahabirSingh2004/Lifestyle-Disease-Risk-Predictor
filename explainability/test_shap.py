# ============================================================
# SHAP EXPLAINABILITY TEST
# Lifestyle Disease Risk Predictor + Habit Simulator
# ============================================================

import os
import matplotlib.pyplot as plt

from explainability.shap_explainer import (
    explain_diabetes,
    explain_heart_disease,
    explain_sleep_disorder,
    create_shap_waterfall_plot,
    create_shap_bar_plot
)


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR = "shap_outputs"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# COMMON RESULT DISPLAY
# ============================================================

def display_shap_result(
    disease_name,
    result,
    file_prefix
):

    print("\n" + "=" * 60)
    print(f"{disease_name.upper()} SHAP EXPLANATION")
    print("=" * 60)

    print(
        f"Base Probability: "
        f"{result['base_value'] * 100:.2f}%"
    )

    print(
        f"Predicted Probability: "
        f"{result['prediction_probability'] * 100:.2f}%"
    )

    print(
        f"Reconstructed Probability: "
        f"{result['reconstructed_probability'] * 100:.2f}%"
    )

    print(
        f"Additivity Difference: "
        f"{result['additivity_difference']:.8f}"
    )

    print("\nTop 5 Contributing Features:")

    for item in result["feature_contributions"][:5]:

        print(
            f"{item['feature']}: "
            f"SHAP = {item['shap_value']:.5f} | "
            f"{item['direction']}"
        )

    # --------------------------------------------------------
    # Waterfall plot
    # --------------------------------------------------------

    waterfall_fig = create_shap_waterfall_plot(
        result
    )

    waterfall_path = os.path.join(
        OUTPUT_DIR,
        f"{file_prefix}_waterfall.png"
    )

    waterfall_fig.savefig(
        waterfall_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(waterfall_fig)

    # --------------------------------------------------------
    # Local bar plot
    # --------------------------------------------------------

    bar_fig = create_shap_bar_plot(
        result
    )

    bar_path = os.path.join(
        OUTPUT_DIR,
        f"{file_prefix}_bar.png"
    )

    bar_fig.savefig(
        bar_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(bar_fig)

    print("\nPlots saved:")
    print(waterfall_path)
    print(bar_path)


# ============================================================
# 1. DIABETES SHAP TEST
# ============================================================

diabetes_result = explain_diabetes(
    gender="Male",
    age=66,
    hypertension=1,
    heart_disease=0,
    smoking_history="former",
    bmi=41.49,
    hba1c_level=6.6,
    blood_glucose_level=158
)

display_shap_result(
    "Diabetes",
    diabetes_result,
    "diabetes"
)


# ============================================================
# 2. HEART DISEASE SHAP TEST
# ============================================================

heart_result = explain_heart_disease(
    age=75,
    gender="Male",
    cholesterol=300,
    blood_pressure=160,
    heart_rate=90,
    smoking="Current",
    alcohol_intake="Heavy",
    exercise_hours=0,
    family_history="Yes",
    diabetes="Yes",
    obesity="Yes",
    stress_level=9,
    blood_sugar=180,
    exercise_induced_angina="Yes",
    chest_pain_type="Typical Angina"
)

display_shap_result(
    "Heart Disease",
    heart_result,
    "heart_disease"
)


# ============================================================
# 3. SLEEP DISORDER SHAP TEST
# ============================================================

sleep_result = explain_sleep_disorder(
    person_id=1,
    gender="Male",
    age=55,
    occupation="Salesperson",
    sleep_duration=5,
    quality_of_sleep=3,
    physical_activity_level=20,
    stress_level=8,
    bmi_category="Obese",
    blood_pressure="140/95",
    heart_rate=90,
    daily_steps=3000
)

display_shap_result(
    "Sleep Disorder",
    sleep_result,
    "sleep_disorder"
)


# ============================================================
# COMPLETION
# ============================================================

print("\n" + "=" * 60)
print("ALL SHAP TESTS COMPLETED")
print("=" * 60)

print(
    f"\nAll plots are available in the '{OUTPUT_DIR}' folder."
)