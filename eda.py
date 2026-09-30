"""
eda.py
------
Exploratory Data Analysis (EDA): look at the data with charts before modelling.
All charts are saved as PNG files inside the results/ folder.

Run:
    python eda.py
"""

import matplotlib
matplotlib.use("Agg")  # save charts to files without opening windows
import matplotlib.pyplot as plt
import seaborn as sns

import config
from preprocessing import load_raw_data, clean_data


def save_figure(filename):
    """Save the current chart into results/ and close it."""
    plt.tight_layout()
    plt.savefig(config.RESULTS_DIR / filename, dpi=120)
    plt.close()
    print("Saved:", filename)


def add_label_column(df):
    """Add a readable text column so charts say 'Diabetic' instead of 1."""
    df = df.copy()
    df["Diagnosis"] = df[config.TARGET_COLUMN].map({0: "Non-diabetic", 1: "Diabetic"})
    return df


def plot_class_distribution(df):
    """How many diabetic vs non-diabetic patients? (shows class imbalance)"""
    counts = df["Diagnosis"].value_counts()
    plt.figure(figsize=(5, 4))
    ax = sns.barplot(x=counts.index, y=counts.values)
    for i, value in enumerate(counts.values):
        ax.text(i, value, str(value), ha="center", va="bottom")
    plt.title("Class Distribution")
    plt.ylabel("Number of patients")
    plt.xlabel("")
    save_figure("class_distribution.png")


def plot_missing_values(df):
    """How many values became missing after treating zeros as invalid?"""
    missing = df[config.FEATURE_COLUMNS].isnull().sum()
    plt.figure(figsize=(7, 4))
    sns.barplot(x=missing.values, y=missing.index)
    plt.title("Missing Values After Handling Invalid Zeros")
    plt.xlabel("Number of missing values")
    save_figure("missing_values.png")


def plot_histograms(df):
    """Distribution of every feature, split by diagnosis."""
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for ax, feature in zip(axes.flatten(), config.FEATURE_COLUMNS):
        sns.histplot(data=df, x=feature, hue="Diagnosis", element="step",
                     stat="density", common_norm=False, ax=ax)
        ax.set_title(feature)
    save_figure("feature_distributions.png")


def plot_boxplots(df):
    """Boxplots help spot outliers and differences between the two groups."""
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for ax, feature in zip(axes.flatten(), config.FEATURE_COLUMNS):
        sns.boxplot(data=df, x="Diagnosis", y=feature, ax=ax)
        ax.set_title(feature)
        ax.set_xlabel("")
    save_figure("boxplots.png")


def plot_correlation_heatmap(df):
    """How strongly are features related to each other and to the Outcome?"""
    columns = config.FEATURE_COLUMNS + [config.TARGET_COLUMN]
    correlation = df[columns].corr()
    plt.figure(figsize=(9, 7))
    sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap")
    save_figure("correlation_heatmap.png")

    # Also save the correlation of each feature with the Outcome as a table
    with_outcome = correlation[config.TARGET_COLUMN].drop(config.TARGET_COLUMN)
    with_outcome = with_outcome.sort_values(ascending=False)
    with_outcome.to_csv(config.RESULTS_DIR / "correlation_with_outcome.csv",
                        header=["correlation"])
    print("\nCorrelation of each feature with Outcome:")
    print(with_outcome.round(3).to_string())


if __name__ == "__main__":
    config.create_output_folders()
    data = add_label_column(clean_data(load_raw_data()))
    plot_class_distribution(data)
    plot_missing_values(data)
    plot_histograms(data)
    plot_boxplots(data)
    plot_correlation_heatmap(data)
    print("\nEDA finished. Charts are in the results/ folder.")
