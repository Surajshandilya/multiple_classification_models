"""Interactive comparison of five saved purchase-intent classifiers.

Upload a test CSV that still has the original session columns and the Revenue
label. The app loads the pipelines from model/ and scores them. It does not
retrain.
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from evaluation import (
    METRIC_NAMES,
    compute_classification_metrics,
    confusion_counts,
    positive_class_scores,
    purchase_classification_report,
)
from model_catalog import MODEL_CATALOG
from preprocessing import FEATURE_COLUMNS, NUMERIC_FEATURES, TARGET_COLUMN, purchase_labels

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "model"
BUNDLED_TEST = PROJECT_DIR / "test_data.csv"

MODEL_NOTES = {
    "Logistic Regression": (
        "Linear model with balanced class weights. Coefficients describe how "
        "page value, exit rate, and session context push a visit toward a purchase."
    ),
    "Decision Tree": (
        "Shallow balanced tree. Splits stay in the original units of bounce rate, "
        "page value, and page counts, which makes individual rules easier to read."
    ),
    "KNN": (
        "Labels a session from the nearest standardized training visits. "
        "Distance weighting gives closer sessions more influence than far ones."
    ),
    "Naive Bayes": (
        "Multinomial Naive Bayes on the raw non-negative counts, durations, rates, "
        "and one-hot codes. Gaussian Naive Bayes is a poor match once browser, month, "
        "and traffic type are expanded into many binary columns."
    ),
    "Support Vector Machine": (
        "Linear SVM (LinearSVC) with balanced class weights. A linear kernel keeps "
        "the saved model small enough for Streamlit Community Cloud. AUC uses the "
        "signed decision value of the purchase class."
    ),
}


# Method: load_pipelines
# Description: Loads the five saved classifier pipelines from the model folder once per app session.
# Params: none
@st.cache_resource
def load_pipelines():
    loaded = []
    for spec in MODEL_CATALOG:
        path = MODEL_DIR / spec["filename"]
        loaded.append((spec["name"], joblib.load(path)))
    return loaded


# Method: read_test_frame
# Description: Reads the uploaded test CSV, or the bundled held-out test_data.csv when nothing was uploaded.
# Params: uploaded_file (UploadedFile or None) - CSV from the sidebar uploader
def read_test_frame(uploaded_file):
    if uploaded_file is None:
        return pd.read_csv(BUNDLED_TEST), "Bundled held-out test file (test_data.csv)"
    return pd.read_csv(uploaded_file), f"Uploaded file ({uploaded_file.name})"


# Method: prepare_sessions
# Description: Checks required columns, parses numeric fields, and maps Revenue to 0 or 1.
# Params: frame (DataFrame) - raw test sessions including the Revenue column
def prepare_sessions(frame: pd.DataFrame):
    required = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(
            "The CSV must include every original input feature and the Revenue column. "
            f"Missing: {', '.join(missing)}"
        )
    working = frame[required].copy()
    for column in NUMERIC_FEATURES:
        working[column] = pd.to_numeric(working[column], errors="coerce")
    dropped = int(working[NUMERIC_FEATURES + [TARGET_COLUMN]].isna().any(axis=1).sum())
    working = working.dropna(subset=NUMERIC_FEATURES + [TARGET_COLUMN])
    if working.empty:
        raise ValueError("No usable rows remain after reading the numeric columns and Revenue.")
    labels = purchase_labels(working[TARGET_COLUMN])
    return working[FEATURE_COLUMNS], labels, dropped


# Method: score_pipeline
# Description: Predicts with a saved pipeline and returns purchase-class metrics without refitting.
# Params: estimator (Pipeline) - fitted preprocessor and classifier
# features (DataFrame) - original session columns
# labels (Series) - purchase labels, 1 for a purchase and 0 otherwise
def score_pipeline(estimator, features, labels):
    predictions = estimator.predict(features)
    scores = positive_class_scores(estimator, features)
    metrics = compute_classification_metrics(labels, predictions, scores)
    return predictions, metrics


# Method: render_metric_cards
# Description: Shows Accuracy, AUC, Precision, Recall, F1, and MCC for the selected model.
# Params: metrics (dict) - metric name mapped to its score on the current test file
def render_metric_cards(metrics):
    columns = st.columns(len(METRIC_NAMES))
    for column, name in zip(columns, METRIC_NAMES):
        column.metric(name, f"{metrics[name]:.4f}")


# Method: render_confusion_matrix
# Description: Draws the 2 by 2 confusion matrix for no-purchase versus purchase.
# Params: labels (Series) - actual purchase labels
# predictions (ndarray) - labels predicted by the selected model
# model_name (str) - display name of the selected classifier
def render_confusion_matrix(labels, predictions, model_name):
    counts = confusion_counts(labels, predictions)
    fig, axis = plt.subplots(figsize=(5.2, 4.2))
    sns.heatmap(
        counts,
        annot=True,
        fmt="d",
        cmap="GnBu",
        cbar=False,
        xticklabels=["No purchase", "Purchase"],
        yticklabels=["No purchase", "Purchase"],
        ax=axis,
    )
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    axis.set_title(f"{model_name} on the current test file")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# Method: render_report
# Description: Shows the per-class classification report for the selected model.
# Params: labels (Series) - actual purchase labels
# predictions (ndarray) - labels predicted by the selected model
def render_report(labels, predictions):
    report = purchase_classification_report(labels, predictions)
    rows = []
    for label, values in report.items():
        if isinstance(values, dict):
            rows.append({"Label": label, **values})
        else:
            rows.append({"Label": label, "f1-score": values})
    report_frame = pd.DataFrame(rows)
    numeric_columns = [column for column in report_frame.columns if column != "Label"]
    styled = report_frame.style.format({column: "{:.4f}" for column in numeric_columns}, na_rep="")
    st.dataframe(styled, width="stretch", hide_index=True)


# Method: inject_styles
# Description: Applies the page colors and metric-card layout used by the Streamlit app.
# Params: none
def inject_styles():
    st.markdown(
        """
        <style>
        .stApp { background: #f3f6f4; }
        h1 { color: #143d36; letter-spacing: -0.4px; }
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #d5e4dc;
            border-left: 4px solid #1f7a63;
            padding: 0.6rem 0.8rem;
            border-radius: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# Method: main
# Description: Builds the upload control, model dropdown, metric table, confusion matrix, and report.
# Params: none
def main():
    st.set_page_config(
        page_title="Shopper Purchase Classifier",
        page_icon=":shopping_cart:",
        layout="wide",
    )
    inject_styles()
    st.title("Will this visit end in a purchase?")
    st.write(
        "Five classifiers trained on the UCI Online Shoppers Purchasing Intention "
        "dataset. Choose a saved model, keep the bundled test sessions or upload "
        "your own CSV, and read accuracy, AUC, precision, recall, F1, and MCC. "
        "Nothing on this page refits a model."
    )

    pipelines = load_pipelines()
    names = [name for name, _ in pipelines]
    with st.sidebar:
        st.header("Evaluation setup")
        uploaded = st.file_uploader(
            "Upload test CSV",
            type=["csv"],
            help="Include the 17 input features and the Revenue column. Training rows are not required.",
        )
        selected_name = st.selectbox("Model", names, index=0)
        st.caption(
            "Leave the uploader empty to score the held-out test_data.csv shipped with this app."
        )

    try:
        raw_frame, source_label = read_test_frame(uploaded)
        features, labels, dropped = prepare_sessions(raw_frame)
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    purchase_count = int(labels.sum())
    st.caption(
        f"{source_label} · {len(labels)} sessions · {purchase_count} purchases "
        f"({purchase_count / len(labels):.1%})"
        + (f" · {dropped} rows dropped for missing numeric values" if dropped else "")
    )

    metric_rows = []
    selected = None
    try:
        for name, estimator in pipelines:
            predictions, metrics = score_pipeline(estimator, features, labels)
            metric_rows.append({"ML Model Name": name, **metrics})
            if name == selected_name:
                selected = (predictions, metrics)
    except ValueError as exc:
        st.error(
            "Metrics could not be computed. The file needs both purchases and "
            f"non-purchases so AUC and MCC are defined. Details: {exc}"
        )
        st.stop()

    comparison = pd.DataFrame(metric_rows)
    st.subheader("Metrics on the current test file")
    st.dataframe(
        comparison.style.format({name: "{:.4f}" for name in METRIC_NAMES}),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "Precision, recall, and F1 refer to the purchase class. AUC uses the "
        "model's purchase score. MCC summarizes all four confusion-matrix cells."
    )

    predictions, metrics = selected
    st.subheader(selected_name)
    st.write(MODEL_NOTES[selected_name])
    render_metric_cards(metrics)

    matrix_tab, report_tab, preview_tab = st.tabs(
        ["Confusion matrix", "Classification report", "Test rows"]
    )
    with matrix_tab:
        render_confusion_matrix(labels, predictions, selected_name)
    with report_tab:
        render_report(labels, predictions)
    with preview_tab:
        preview = features.head(12).copy()
        preview[TARGET_COLUMN] = labels.head(12).to_numpy()
        st.dataframe(preview, width="stretch", hide_index=True)


if __name__ == "__main__":
    main()
