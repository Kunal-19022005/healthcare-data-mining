# Healthcare Data Mining for Disease Diagnosis

Diabetes risk prediction using classification techniques: a B.Tech Data Mining mini-project (AI & Data Science).

> **Disclaimer:** This project is for educational purposes only and must not be used as a substitute for professional medical diagnosis.

## Problem Statement
Hospitals store many patient measurements (glucose, BMI, age, etc.), but the patterns linking them to diabetes are hard to see by eye. This project uses data mining to find those patterns and to build classification models that estimate whether a patient is at higher risk of diabetes.

## Objectives
- Clean healthcare data, including physiologically invalid values
- Explore the data and find which features relate to diabetes
- Train and compare five classification algorithms
- Evaluate with accuracy, precision, recall, F1-score and confusion matrix
- Save the best model and use it in a simple Streamlit web app

## Dataset
**Pima Indians Diabetes Database** (originally from the National Institute of Diabetes and Digestive and Kidney Diseases; available on UCI and Kaggle). Check the licence on the page you download it from before re-uploading the CSV.

- Input features: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
- Target: `Outcome` (1 = diabetic, 0 = non-diabetic)
- Exact row counts and class counts are printed by `python preprocessing.py`.

**Key data-quality issue:** columns such as Glucose, BloodPressure, SkinThickness, Insulin and BMI contain zeros that are impossible for a living person. They are treated as missing values.

## Technologies
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Streamlit, Joblib

## Data Mining Concepts Used
Data cleaning, missing-value imputation (median), feature scaling, exploratory data analysis, correlation analysis, feature importance, stratified train-test split, cross-validation, supervised classification, model evaluation and comparison.

## Methodology / Workflow
```
Raw CSV -> Cleaning (duplicates, invalid zeros -> NaN)
        -> Stratified 80/20 train-test split
        -> Pipeline: median imputation -> scaling -> classifier
        -> Evaluation on unseen test data (+ 5-fold cross-validation)
        -> Best model saved -> Streamlit app
```
The imputer and scaler are part of the scikit-learn Pipeline, so they learn only from training data. This avoids data leakage.

## Algorithms
| Algorithm | Why it is used |
|---|---|
| Logistic Regression | Simple, interpretable baseline that outputs probabilities |
| Decision Tree | Human-readable if-then rules (depth limited to reduce overfitting) |
| K-Nearest Neighbors | Instance-based method, contrasts with model-based ones |
| Naive Bayes | Fast probabilistic model; assumes features are independent |
| Random Forest | Ensemble of trees; usually strong and gives feature importance |

## Evaluation Metrics
- **Accuracy:** share of all predictions that are correct
- **Precision:** of patients predicted diabetic, how many really are
- **Recall:** of truly diabetic patients, how many were caught (most important in healthcare)
- **F1-score:** balance of precision and recall
- **Confusion matrix:** counts of TN, FP, FN, TP

**Best model rule:** highest F1-score; ties are broken by recall.

## Project Structure
```
healthcare-data-mining/
├── config.py            # paths and constants
├── preprocessing.py     # load, clean, split
├── eda.py               # charts -> results/
├── train.py             # trains 5 models -> models/
├── evaluate.py          # metrics, charts, saves best model
├── predict.py           # prediction function used by the app
├── app.py               # Streamlit web app
├── data/diabetes.csv
├── models/
├── results/
├── screenshots/         # add your app screenshots here
├── requirements.txt
└── README.md
```

## Installation
```bash
python -m venv venv
venv\Scripts\activate          # Windows   (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
```

## How to Run (in this order)
```bash
python preprocessing.py
python eda.py
python train.py
python evaluate.py
python predict.py
streamlit run app.py
```

## Results
Run `python evaluate.py`, then copy your own numbers from `results/model_comparison.csv` into this table.

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | | | | |
| Decision Tree | | | | |
| KNN | | | | |
| Naive Bayes | | | | |
| Random Forest | | | | |

Charts are saved in `results/` (confusion matrices, comparison chart, feature importance).

## Screenshots
Add screenshots of the app pages to `screenshots/` and link them here.

## Example Prediction
Run `python predict.py` (it uses a sample patient) and paste the output here.

## Limitations
- Small dataset (few hundred patients), all female Pima Indian patients, so results may not generalise
- Zeros were treated as missing and filled with the median, which is an approximation
- Only a few features; no clinical history or lifestyle data
- Predicted probabilities are not calibrated medical risk scores
- Educational project, not validated for clinical use

## Future Scope
Hyperparameter tuning, handling class imbalance with SMOTE, SHAP explanations, larger multi-disease datasets, deployment on Streamlit Community Cloud.

## Medical Disclaimer
This application is developed for educational purposes only and should not be used as a substitute for professional medical diagnosis.
