# Lifestyle Disease Risk Predictor with Habit Simulator and SHAP Explainability

An AI-powered web application that predicts the probability of lifestyle-related diseases using Machine Learning. The application provides predictions for Diabetes, Heart Disease, and Sleep Disorders, along with a Habit Simulator and SHAP-based explainability to help users understand model predictions.

## Project Overview

Lifestyle-related diseases are influenced by several health and lifestyle factors, including age, BMI, blood glucose, physical activity, sleep duration, stress, and other health indicators.

This project uses Machine Learning algorithms to analyse these factors and estimate disease probabilities. It also provides an interactive Habit Simulator that allows users to modify selected lifestyle inputs and compare the resulting model predictions.

SHAP (SHapley Additive exPlanations) is integrated to explain how individual input features contribute to a model's prediction.

## Key Features

- **Diabetes Risk Prediction:** Estimates the probability of diabetes using health indicators such as age, BMI, HbA1c level, blood glucose, hypertension, and smoking history.
- **Heart Disease Risk Prediction:** Estimates the probability of heart disease using factors such as age, cholesterol, blood pressure, heart rate, smoking, exercise, family history, and other health indicators.
- **Sleep Disorder Prediction:** Predicts sleep disorder-related outcomes using sleep duration, sleep quality, physical activity, stress level, BMI category, heart rate, and other lifestyle information.
- **Habit Simulator:** Allows users to modify selected lifestyle inputs and compare original and simulated model predictions.
- **SHAP Explainability:** Identifies features that contribute positively or negatively to an individual model prediction.
- **Interactive Dashboard:** Provides a user-friendly interface through Streamlit.
- **Visual Explanations:** Displays SHAP waterfall and bar plots to make model behaviour easier to understand.

## Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Frontend | Streamlit |
| Machine Learning | Scikit-learn |
| Data Processing | Pandas, NumPy |
| Data Preprocessing | StandardScaler, Categorical Encoding |
| Explainable AI | SHAP |
| Data Visualization | Matplotlib |
| Model Storage | Pickle |
| Development Environment | VS Code, Jupyter Notebook / Google Colab |

## Machine Learning Algorithms

The project explored the following classification algorithms during model development:

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
4. Gaussian Naive Bayes

The models were evaluated using classification and probability-based evaluation metrics. The trained models and preprocessing objects are saved for use by the prediction application.

## Project Structure

```text
Lifestyle-Disease-Risk-Predictor/
│
├── datasets/
│   └── Dataset files
│
├── models/
│   ├── diabetes_model.pkl
│   ├── diabetes_scaler.pkl
│   ├── heart_model.pkl
│   ├── heart_scaler.pkl
│   ├── sleep_model.pkl
│   ├── sleep_scaler.pkl
│   ├── encoding_maps.pkl
│   └── model_metadata.pkl
│
├── prediction/
│   └── prediction.py
│
├── explainability/
│   ├── shap_explainer.py
│   └── test_shap.py
│
├── frontend/
│   └── app.py
│
├── train_model.py
├── evaluate_models.py
├── requirements.txt
├── README.md
└── .gitignore
```

*Note: This is the main project structure. Additional files may be present in the actual repository.*

## How the System Works

The application follows these steps:

1. **User Input:** The user enters health and lifestyle information through the Streamlit interface.
2. **Data Preprocessing:** Categorical inputs are encoded and numerical values are standardized using the saved preprocessing objects.
3. **Model Prediction:** The processed input is passed to the corresponding trained Machine Learning model.
4. **Probability Calculation:** The model estimates the probability of the positive disease class. The probability is multiplied by 100 to display a percentage.
5. **Habit Simulation:** Users modify selected inputs, and the same trained model generates a new prediction for comparison.
6. **SHAP Explainability:** SHAP calculates feature contributions to help explain the model's individual prediction.
7. **Result Display:** The application displays predictions, comparisons, and visual explanations.

## Model Evaluation

The trained models were evaluated using the following metrics:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Score
- Log Loss
- Brier Score

### Evaluation Results

| Metric | Diabetes | Heart Disease | Sleep Disorder |
|---|---:|---:|---:|
| Accuracy | 97.01% | 86.50% | 94.67% |
| Precision | 93.96% | 84.93% | 96%* |
| Recall | 70.64% | 79.49% | 95%* |
| F1-Score | 80.65% | 82.12% | 95%* |
| ROC-AUC | 0.9703 | 0.9504 | 0.9833 |
| Log Loss | 0.0977 | 0.2773 | 0.1677 |
| Brier Score | 0.0249 | 0.0883 | 0.0420 |

*For Sleep Disorder, the precision, recall, and F1-score shown above are weighted classification metrics. The Sleep Apnea class-specific precision is 78.95%, recall is 100%, and F1-score is 88.24%.*

These results represent evaluation on the project's test datasets. They do not represent clinical validation.

## SHAP Explainability

SHAP is used to understand how individual features influence the model's predictions.

The application generates:

- **Waterfall Plot:** Shows how different input features contribute to an individual prediction relative to the model's baseline.
- **Bar Plot:** Displays the most influential feature contributions for the selected prediction.

The SHAP explanation helps users understand model behaviour. It does not establish that a feature causes a disease.

## Habit Simulator

The Habit Simulator is an important feature of this project.

It allows users to change selected lifestyle and health-related inputs and observe how the model's predicted probability changes.

For example:

- Original model prediction: 75%
- Simulated model prediction: 40%
- Difference: 35 percentage points

The simulator demonstrates how the trained model responds to modified inputs. It does not guarantee that making those changes will produce the same reduction in a person's actual medical risk.

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/Lifestyle-Disease-Risk-Predictor.git
```

### 2. Navigate to the project folder

```bash
cd Lifestyle-Disease-Risk-Predictor
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the virtual environment

**Windows PowerShell:**

```powershell
.venv\Scripts\Activate.ps1
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Streamlit application

```bash
streamlit run frontend/app.py
```

The application should open in your default browser. If it does not, open the local URL displayed in the terminal.

## Explainability Outputs

The SHAP module can be tested using:

```bash
python -m explainability.test_shap
```

The generated visualizations are saved in the `shap_outputs/` directory when the explanation script completes successfully.

## Project Objectives

- Develop a Machine Learning-based lifestyle disease prediction application.
- Estimate disease probabilities using relevant health and lifestyle parameters.
- Provide an interactive Habit Simulator for comparing predictions.
- Integrate SHAP to improve the interpretability of Machine Learning predictions.
- Develop an accessible web interface using Streamlit.
- Demonstrate how explainable Machine Learning can support health-risk awareness.

## Limitations

- Predictions depend on the quality and representativeness of the datasets.
- Model performance may vary for individuals with different characteristics or populations.
- SHAP explains model behaviour, not medical causation.
- Habit Simulator results are model-based comparisons, not guaranteed health improvements.
- The application is not a substitute for professional medical advice, diagnosis, or treatment.

## Future Scope

- Integrate larger and more diverse healthcare datasets.
- Improve model generalization and probability calibration.
- Add personalized lifestyle recommendations based on reliable medical guidelines.
- Develop a user account system to save prediction history.
- Add additional lifestyle-related disease prediction modules.
- Deploy the application on a cloud platform.

## Disclaimer

This project is developed for academic and educational purposes. It is not a clinically validated medical diagnostic system. The displayed probabilities are Machine Learning model estimates and should not be interpreted as a confirmed diagnosis or an individual's actual medical risk. Users should consult qualified healthcare professionals for medical advice.

## Author

**Mahabir Singh**  
Master of Science in Data Science  
Somaiya Vidyavihar University

## Acknowledgements

- Scikit-learn for Machine Learning tools.
- Streamlit for web application development.
- SHAP for explainable AI functionality.
- Python and its data science ecosystem for data processing and model development.
