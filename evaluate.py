"""
evaluate.py
-----------
Tests every trained model on the unseen test set and compares them.

Outputs (all saved in results/):
    model_comparison.csv       accuracy, precision, recall, F1, CV scores, confusion counts
    confusion_matrices.png     one confusion matrix per model
    model_comparison.png       bar chart comparing the models
    feature_importance.csv/png which features Random Forest relies on most
    best_model.txt             name of the best model
and models/best_model.joblib   the best model, used by the web app.

Best model rule: highest F1-score (a balance of precision and recall);
if tied, the higher recall wins, because in healthcare missing a sick
patient is the more serious error.

Run:
    python evaluate.py
"""

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score

import config
from preprocessing import get_train_test_split
from train import build_models, model_file_path


def evaluate_models(X_train, X_test, y_train, y_test):
    """Compute all metrics for every model. Returns a DataFrame."""
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True,
                         random_state=config.RANDOM_STATE)
    rows = []
    matrices = {}

    for name, untrained_pipeline in build_models().items():
        path = model_file_path(name)
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run 'python train.py' first.")
        model = joblib.load(path)
        y_pred = model.predict(X_test)

        # Confusion matrix layout: [[TN, FP], [FN, TP]]
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        matrices[name] = [[tn, fp], [fn, tp]]

        # Cross-validation: average F1 over several train/validation splits
        cv_f1 = cross_val_score(untrained_pipeline, X_train, y_train,
                                cv=cv, scoring="f1")
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred),
            "F1-score": f1_score(y_test, y_pred),
            f"CV F1 (mean of {config.CV_FOLDS} folds)": cv_f1.mean(),
            "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        })

    return pd.DataFrame(rows), matrices


def plot_confusion_matrices(matrices):
    fig, axes = plt.subplots(1, len(matrices), figsize=(4 * len(matrices), 4))
    for ax, (name, matrix) in zip(axes, matrices.items()):
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Non-diabetic", "Diabetic"],
                    yticklabels=["Non-diabetic", "Diabetic"])
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(config.RESULTS_DIR / "confusion_matrices.png", dpi=120)
    plt.close()


def plot_model_comparison(results):
    metrics = ["Accuracy", "Precision", "Recall", "F1-score"]
    long_form = results.melt(id_vars="Model", value_vars=metrics,
                             var_name="Metric", value_name="Score")
    plt.figure(figsize=(10, 5))
    sns.barplot(data=long_form, x="Model", y="Score", hue="Metric")
    plt.ylim(0, 1)
    plt.title("Model Comparison on the Test Set")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(config.RESULTS_DIR / "model_comparison.png", dpi=120)
    plt.close()


def save_feature_importance():
    """Random Forest tells us how much each feature helped its decisions."""
    forest = joblib.load(model_file_path("Random Forest")).named_steps["model"]
    importance = pd.Series(forest.feature_importances_, index=config.FEATURE_COLUMNS)
    importance = importance.sort_values(ascending=False)
    importance.to_csv(config.RESULTS_DIR / "feature_importance.csv",
                      header=["importance"], index_label="feature")

    plt.figure(figsize=(7, 4))
    sns.barplot(x=importance.values, y=importance.index)
    plt.title("Feature Importance (Random Forest)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(config.RESULTS_DIR / "feature_importance.png", dpi=120)
    plt.close()
    return importance


def save_best_model(results):
    """Pick the best model (highest F1, then recall) and save it."""
    ranked = results.sort_values(["F1-score", "Recall"], ascending=False)
    best_name = ranked.iloc[0]["Model"]
    joblib.dump(joblib.load(model_file_path(best_name)), config.BEST_MODEL_PATH)
    (config.RESULTS_DIR / "best_model.txt").write_text(best_name)
    return best_name


if __name__ == "__main__":
    config.create_output_folders()
    X_train, X_test, y_train, y_test = get_train_test_split()

    results, matrices = evaluate_models(X_train, X_test, y_train, y_test)
    results.to_csv(config.METRICS_PATH, index=False)
    plot_confusion_matrices(matrices)
    plot_model_comparison(results)
    importance = save_feature_importance()
    best_name = save_best_model(results)

    print(results.round(3).to_string(index=False))
    print("\nFeature importance (Random Forest):")
    print(importance.round(3).to_string())
    print(f"\nBest model: {best_name}")
    print(f"Saved to  : {config.BEST_MODEL_PATH}")
    print("Next step : streamlit run app.py")
