"""
app.py
------
Streamlit web application for the project.
Start it with:
    streamlit run app.py

Pages: Home | Disease Prediction | Data Analysis | Model Comparison
"""

import pandas as pd
import streamlit as st

import config
from predict import predict_risk
from preprocessing import clean_data, load_raw_data

st.set_page_config(page_title="Healthcare Data Mining", page_icon="🩺", layout="wide")


# ---------- Helper functions ----------
@st.cache_data
def load_cleaned_data():
    return clean_data(load_raw_data())


def show_image(filename):
    """Show a chart from results/, or a friendly hint if it is missing."""
    path = config.RESULTS_DIR / filename
    if path.exists():
        st.image(str(path))
    else:
        st.warning(f"{filename} not found. Run the scripts first (see README).")


def read_results_csv(filename):
    path = config.RESULTS_DIR / filename
    return pd.read_csv(path) if path.exists() else None


def best_model_name():
    path = config.RESULTS_DIR / "best_model.txt"
    return path.read_text().strip() if path.exists() else "not available yet"


# ---------- Pages ----------
def home_page():
    # ---------- Hero Section ----------
    st.markdown(
        """
        <style>
        .hero {
            padding: 50px 20px 40px 20px;
            text-align: center;
        }

        .hero h1 {
            font-size: 48px;
            margin-bottom: 10px;
        }

        .hero h2 {
            font-size: 25px;
            font-weight: 400;
            margin-bottom: 20px;
        }

        .hero p {
            font-size: 18px;
            max-width: 750px;
            margin: auto;
        }

        .card {
            padding: 25px;
            border-radius: 15px;
            border: 1px solid rgba(128,128,128,0.25);
            text-align: center;
            height: 150px;
        }

        .card h2 {
            margin-bottom: 5px;
        }

        .section {
            margin-top: 45px;
            margin-bottom: 25px;
        }
        </style>

        <div class="hero">
            <h1>🩺 Healthcare AI</h1>
            <h2>Diabetes Risk Assessment</h2>
            <p>
                An AI-based application that analyzes health parameters
                and provides a machine-learning based diabetes risk assessment.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ---------- Quick Stats ----------
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            """
            <div class="card">
                <h2>768</h2>
                <p>Dataset Samples</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div class="card">
                <h2>8</h2>
                <p>Health Features</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            """
            <div class="card">
                <h2>5</h2>
                <p>ML Models</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:
        st.markdown(
            """
            <div class="card">
                <h2>87%</h2>
                <p>Test Accuracy</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    # ---------- How it works ----------
    st.markdown(
        '<div class="section"><h2>How It Works</h2></div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 👤 1. Enter Details")
        st.write(
            "Provide health parameters such as glucose, BMI, age, "
            "blood pressure and other relevant features."
        )

    with col2:
        st.markdown("### 🤖 2. AI Analysis")
        st.write(
            "The trained machine-learning model analyzes the "
            "provided health parameters."
        )

    with col3:
        st.markdown("### 📊 3. Risk Assessment")
        st.write(
            "The application displays the predicted diabetes risk "
            "and model probability."
        )

    # ---------- Project Overview ----------
    st.markdown(
        '<div class="section"><h2>About the Project</h2></div>',
        unsafe_allow_html=True
    )

    st.write(
        "This academic project demonstrates how healthcare data mining "
        "and machine learning can be used for diabetes risk prediction. "
        "The system performs data cleaning, exploratory data analysis, "
        "preprocessing, model training and evaluation before making a prediction."
    )

    # ---------- Models ----------
    st.markdown(
        '<div class="section"><h2>Machine Learning Models</h2></div>',
        unsafe_allow_html=True
    )

    col1, col2, col3, col4, col5 = st.columns(5)

    models = [
        ("📈", "Logistic Regression"),
        ("🌳", "Decision Tree"),
        ("📍", "KNN"),
        ("🧮", "Naive Bayes"),
        ("🌲", "Random Forest"),
    ]

    for column, (icon, model) in zip(
        [col1, col2, col3, col4, col5], models
    ):
        with column:
            st.markdown(f"### {icon}")
            st.caption(model)

    # ---------- Call to Action ----------
    st.markdown(
        '<div class="section"></div>',
        unsafe_allow_html=True
    )

    st.info(
        "👉 Use the **Disease Prediction** option from the sidebar "
        "to enter patient parameters and generate a risk assessment."
    )

    # ---------- Disclaimer ----------
    st.markdown("---")
    st.caption(
        "⚠️ Educational Use Only — This application is developed for "
        "academic purposes and should not be used as a substitute "
        "for professional medical diagnosis."
    )

def prediction_page():
    st.title("Disease Prediction")
    st.warning(config.DISCLAIMER)
    st.write("Enter the patient's details and click the button.")

    col1, col2 = st.columns(2)
    with col1:
        pregnancies = st.number_input("Pregnancies", 0, 20, 2)
        glucose = st.number_input("Glucose (mg/dL)", 40, 300, 120)
        blood_pressure = st.number_input("Blood Pressure (diastolic, mm Hg)", 30, 150, 70)
        skin_thickness = st.number_input("Skin Thickness (mm)", 5, 100, 25)
    with col2:
        insulin = st.number_input("Insulin (mu U/ml)", 10, 900, 100)
        bmi = st.number_input("BMI", 10.0, 70.0, 30.0, step=0.1)
        pedigree = st.number_input("Diabetes Pedigree Function", 0.05, 2.5, 0.40, step=0.01)
        age = st.number_input("Age (years)", 21, 100, 30)

    if st.button("Predict Disease Risk"):
        patient = {
            "Pregnancies": pregnancies, "Glucose": glucose,
            "BloodPressure": blood_pressure, "SkinThickness": skin_thickness,
            "Insulin": insulin, "BMI": bmi,
            "DiabetesPedigreeFunction": pedigree, "Age": age,
        }
        try:
            predicted_class, probability = predict_risk(patient)
        except FileNotFoundError as error:
            st.error(str(error))
            return

        st.markdown("### Result")
        if predicted_class == 1:
            st.error("Predicted class: **Higher risk of diabetes**")
        else:
            st.success("Predicted class: **Lower risk of diabetes**")
        st.write(f"Model's probability of diabetes: **{probability:.1%}**")
        st.progress(min(max(probability, 0.0), 1.0))
        st.caption(f"Model used: {best_model_name()}")

        explain_prediction(patient)
        st.caption(config.DISCLAIMER)


def explain_prediction(patient):
    """Compare the patient's values with the dataset median for top features."""
    importance = read_results_csv("feature_importance.csv")
    if importance is None:
        return
    st.markdown("### Why this result?")
    st.write(
        "The most influential features in the Random Forest model, with this "
        "patient's value next to the typical (median) value in the dataset:"
    )
    data = load_cleaned_data()
    rows = []
    for feature in importance["feature"].head(3):
        rows.append({
            "Feature": feature,
            "Patient value": patient[feature],
            "Dataset median": round(float(data[feature].median()), 2),
        })
    st.table(pd.DataFrame(rows))
    st.caption("Values well above the median for key features such as Glucose "
               "and BMI usually push the prediction towards higher risk.")


def data_analysis_page():
    st.title("Data Analysis")
    data = load_cleaned_data()

    st.markdown("### Dataset statistics (after cleaning)")
    st.write(f"Rows: **{data.shape[0]}** | Columns: **{data.shape[1]}**")
    st.dataframe(data.describe().T.round(2))

    st.markdown("### Class distribution")
    show_image("class_distribution.png")
    st.markdown("### Missing values after handling invalid zeros")
    show_image("missing_values.png")
    st.markdown("### Feature distributions")
    show_image("feature_distributions.png")
    st.markdown("### Boxplots")
    show_image("boxplots.png")
    st.markdown("### Correlation heatmap")
    show_image("correlation_heatmap.png")


def model_comparison_page():
    st.title("Model Comparison")
    results = read_results_csv("model_comparison.csv")
    if results is None:
        st.warning("No results yet. Run 'python evaluate.py' first.")
        return

    st.success(f"Best model (highest F1, then recall): **{best_model_name()}**")
    st.markdown("### Metrics on the test set")
    st.dataframe(results.round(3))
    st.markdown("### Comparison chart")
    show_image("model_comparison.png")
    st.markdown("### Confusion matrices")
    st.caption("Rows = actual class, columns = predicted class.")
    show_image("confusion_matrices.png")
    st.markdown("### Feature importance")
    show_image("feature_importance.png")


# ---------- Navigation ----------
page = st.sidebar.radio(
    "Go to", ["Home", "Disease Prediction", "Data Analysis", "Model Comparison"])

if page == "Home":
    home_page()
elif page == "Disease Prediction":
    prediction_page()
elif page == "Data Analysis":
    data_analysis_page()
else:
    model_comparison_page()
