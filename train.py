"""
train.py
--------
Trains five classification models and saves each one inside models/.

Every model is wrapped in a scikit-learn Pipeline with three steps:
    1. SimpleImputer   -> fills missing values with the median
    2. StandardScaler  -> puts all features on a similar scale
    3. The classifier  -> learns from the data

Why a Pipeline? The imputer and scaler are learned ONLY from training data,
and are applied automatically to new data. This prevents data leakage and
means the web app can pass raw values straight to the model.

Run:
    python train.py
"""

import joblib
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

import config
from preprocessing import get_train_test_split


def build_models():
    """Return a dictionary of {model name: untrained pipeline}."""
    classifiers = {
        # class_weight="balanced" gives more importance to the smaller
        # (diabetic) class, which helps recall.
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=config.RANDOM_STATE),
        # max_depth limits tree size so it does not memorise (overfit) the data
        "Decision Tree": DecisionTreeClassifier(
            max_depth=4, class_weight="balanced", random_state=config.RANDOM_STATE),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=config.RANDOM_STATE),
    }

    models = {}
    for name, classifier in classifiers.items():
        models[name] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", classifier),
        ])
    return models


def model_file_path(model_name):
    """e.g. 'Random Forest' -> models/random_forest.joblib"""
    filename = model_name.lower().replace(" ", "_") + ".joblib"
    return config.MODELS_DIR / filename


if __name__ == "__main__":
    config.create_output_folders()
    X_train, X_test, y_train, y_test = get_train_test_split()
    print(f"Training rows: {len(X_train)} | Testing rows: {len(X_test)}\n")

    for name, pipeline in build_models().items():
        pipeline.fit(X_train, y_train)            # learning happens here
        joblib.dump(pipeline, model_file_path(name))
        print(f"Trained and saved: {name}")

    print("\nTraining finished. Next step: python evaluate.py")
