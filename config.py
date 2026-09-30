"""
config.py
---------
Central place for all paths and constants used in the project.
Every other file imports from here, so paths are defined only once.
"""

from pathlib import Path

# ---------- Folder paths ----------
# This file lives in <project>/src/, so the project root is one level up.
PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# ---------- File paths ----------
RAW_DATA_PATH = DATA_DIR / "diabetes.csv"
CLEANED_DATA_PATH = DATA_DIR / "diabetes_cleaned.csv"
METRICS_PATH = RESULTS_DIR / "model_comparison.csv"
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"

# ---------- Dataset column names ----------
TARGET_COLUMN = "Outcome"  # 1 = diabetic, 0 = non-diabetic

FEATURE_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

# In this dataset, a value of 0 is impossible for these columns
# (e.g. a living person cannot have 0 blood pressure), so a 0 means "missing".
# Pregnancies is NOT listed here because 0 pregnancies is a valid value.
INVALID_ZERO_COLUMNS = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]

# ---------- Experiment settings ----------
RANDOM_STATE = 42  # fixed seed so results are reproducible
TEST_SIZE = 0.2    # 80% training, 20% testing
CV_FOLDS = 5       # number of folds for cross-validation

# ---------- Shown in the app ----------
DISCLAIMER = (
    "This application is developed for educational purposes only and "
    "should not be used as a substitute for professional medical diagnosis."
)


def create_output_folders():
    """Make sure the output folders exist before saving anything."""
    for folder in (DATA_DIR, MODELS_DIR, RESULTS_DIR):
        folder.mkdir(parents=True, exist_ok=True)
