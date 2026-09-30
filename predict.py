"""
predict.py
----------
Loads the saved best model and predicts diabetes risk for ONE patient.
Used by the Streamlit app (app.py), but can also be tested alone:
    python predict.py
"""

import pandas as pd
import joblib

import config


def load_best_model():
    if not config.BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            "No trained model found. Run 'python train.py' and "
            "'python evaluate.py' first."
        )
    return joblib.load(config.BEST_MODEL_PATH)


def predict_risk(patient):
    """
    patient: dictionary such as {"Glucose": 120, "BMI": 28.0, ...}
    Returns (predicted_class, probability_of_diabetes)
        predicted_class: 1 = higher risk of diabetes, 0 = lower risk
    """
    model = load_best_model()
    # One-row table with columns in the same order used during training
    patient_table = pd.DataFrame([patient])[config.FEATURE_COLUMNS]
    predicted_class = int(model.predict(patient_table)[0])
    probability = float(model.predict_proba(patient_table)[0][1])
    return predicted_class, probability


if __name__ == "__main__":
    sample_patient = {
        "Pregnancies": 2, "Glucose": 150, "BloodPressure": 80,
        "SkinThickness": 30, "Insulin": 130, "BMI": 33.0,
        "DiabetesPedigreeFunction": 0.6, "Age": 45,
    }
    label, prob = predict_risk(sample_patient)
    print("Predicted class:", label, "(1 = higher risk, 0 = lower risk)")
    print(f"Probability of diabetes: {prob:.2%}")
