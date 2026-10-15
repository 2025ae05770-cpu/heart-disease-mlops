"""Exploratory data analysis: generates figures + summary tables into reports/.

Usage:
    python -m src.eda.run_eda
"""

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

from src.config import (  # noqa: E402
    CATEGORICAL_FEATURES,
    FIGURES_DIR,
    NUMERIC_FEATURES,
    REPORTS_DIR,
    TARGET_COLUMN,
)
from src.data.clean_data import load_clean_data  # noqa: E402

logger = logging.getLogger(__name__)

PALETTE = {0: "#2a9d8f", 1: "#e76f51"}
TARGET_LABELS = {0: "No disease", 1: "Disease"}
CATEGORY_LABELS = {
    "sex": {0: "Female", 1: "Male"},
    "cp": {1: "Typical angina", 2: "Atypical angina", 3: "Non-anginal", 4: "Asymptomatic"},
    "fbs": {0: "<=120 mg/dl", 1: ">120 mg/dl"},
    "restecg": {0: "Normal", 1: "ST-T abnormal", 2: "LV hypertrophy"},
    "exang": {0: "No", 1: "Yes"},
    "slope": {1: "Upsloping", 2: "Flat", 3: "Downsloping"},
    "thal": {3: "Normal", 6: "Fixed defect", 7: "Reversible"},
}


def _save(fig, name, output_dir):
    path = Path(output_dir) / name
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_class_balance(data, output_dir):
    counts = data[TARGET_COLUMN].map(TARGET_LABELS).value_counts()
    fig, ax = plt.subplots(figsize=(5.5, 4))
    bars = ax.bar(counts.index, counts.values, color=[PALETTE[0], PALETTE[1]])
    for bar, value in zip(bars, counts.values):
        ax.annotate(f"{value} ({value / counts.sum():.0%})", (bar.get_x() + bar.get_width() / 2, value),
                    ha="center", va="bottom", fontsize=10)
    ax.set_title("Class balance of the target")
    ax.set_ylabel("Patients")
    return _save(fig, "01_class_balance.png", output_dir)


def plot_numeric_histograms(data, output_dir):
    frame = data.assign(diagnosis=data[TARGET_COLUMN].map(TARGET_LABELS))
    colours = {TARGET_LABELS[k]: v for k, v in PALETTE.items()}
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    for ax, column in zip(axes.ravel(), NUMERIC_FEATURES):
        sns.histplot(data=frame, x=column, hue="diagnosis", kde=True, palette=colours,
                     element="step", stat="density", common_norm=False, ax=ax)
        ax.set_title(f"Distribution of {column}")
    fig.suptitle("Numeric feature distributions by diagnosis", fontsize=14)
    return _save(fig, "02_numeric_histograms.png", output_dir)


def plot_categorical_rates(data, output_dir):
    fig, axes = plt.subplots(2, 4, figsize=(17, 7.5))
    for ax, column in zip(axes.ravel(), CATEGORICAL_FEATURES):
        frame = data[[column, TARGET_COLUMN]].dropna()
        rate = frame.groupby(column)[TARGET_COLUMN].agg(["mean", "count"]).reset_index()
        labels = CATEGORY_LABELS.get(column, {})
        rate["label"] = [labels.get(int(v), str(v)) for v in rate[column]]
        sns.barplot(data=rate, x="label", y="mean", color="#457b9d", ax=ax)
        for patch, n in zip(ax.patches, rate["count"]):
            ax.annotate(f"n={n}", (patch.get_x() + patch.get_width() / 2, patch.get_height()),
                        ha="center", va="bottom", fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_title(f"Disease rate by {column}")
        ax.set_xlabel("")
        ax.set_ylabel("P(disease)")
        ax.tick_params(axis="x", rotation=25, labelsize=8)
    axes.ravel()[-1].axis("off")
    fig.suptitle("Observed disease rate per categorical level", fontsize=14)
    return _save(fig, "03_categorical_disease_rate.png", output_dir)


def plot_correlation_heatmap(data, output_dir):
    corr = data.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, square=True,
                cbar_kws={"shrink": 0.8}, annot_kws={"size": 8}, ax=ax)
    ax.set_title("Pearson correlation matrix (features + target)")
    return _save(fig, "04_correlation_heatmap.png", output_dir), corr


def plot_boxplots(data, output_dir):
    long = data.melt(id_vars=TARGET_COLUMN, value_vars=NUMERIC_FEATURES)
    long["diagnosis"] = long[TARGET_COLUMN].map(TARGET_LABELS)
    colours = {TARGET_LABELS[k]: v for k, v in PALETTE.items()}
    g = sns.catplot(data=long, x="diagnosis", y="value", col="variable", kind="box", col_wrap=3,
                    sharey=False, height=3.2, aspect=1.2, palette=colours, hue="diagnosis", legend=False)
    g.set_titles("{col_name}")
    g.set_axis_labels("", "")
    g.figure.suptitle("Outliers and spread of numeric features by diagnosis", y=1.03)
    path = Path(output_dir) / "05_boxplots.png"
    g.figure.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(g.figure)
    return path


def plot_missing_values(data, output_dir):
    missing = data.isna().sum()
    missing = missing[missing > 0]
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    if missing.empty:
        ax.text(0.5, 0.5, "No missing values", ha="center", va="center")
        ax.axis("off")
    else:
        ax.bar(missing.index, missing.values, color="#f4a261")
        ax.set_ylabel("Missing count")
        ax.set_title("Missing values per column (imputed in pipeline)")
    return _save(fig, "06_missing_values.png", output_dir)


def run_eda(data=None, output_dir=FIGURES_DIR, reports_dir=REPORTS_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if data is None:
        data = load_clean_data()

    sns.set_theme(style="whitegrid")
    figures = [
        plot_class_balance(data, output_dir),
        plot_numeric_histograms(data, output_dir),
        plot_categorical_rates(data, output_dir),
    ]
    heatmap_path, corr = plot_correlation_heatmap(data, output_dir)
    figures += [heatmap_path, plot_boxplots(data, output_dir), plot_missing_values(data, output_dir)]

    summary = {
        "shape": list(data.shape),
        "class_balance": {str(k): int(v) for k, v in data[TARGET_COLUMN].value_counts().items()},
        "missing_values": {k: int(v) for k, v in data.isna().sum().items() if v},
        "target_correlation": corr[TARGET_COLUMN].drop(TARGET_COLUMN).sort_values(
            key=abs, ascending=False).round(3).to_dict(),
    }
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    data.describe().T.round(2).to_csv(reports_dir / "eda_summary_statistics.csv")
    (reports_dir / "eda_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("EDA figures written to %s", output_dir)
    return figures, summary


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    figures, summary = run_eda()
    print(json.dumps(summary, indent=2))
    for path in figures:
        print(f"saved {path}")


if __name__ == "__main__":
    main()
