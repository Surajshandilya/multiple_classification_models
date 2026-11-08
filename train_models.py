"""Train the five purchase classifiers and write the held-out test file.

The test sessions are written to test_data.csv before any model sees them.
Hyperparameters are chosen with stratified cross-validation on the training
split only. Saved objects are full pipelines (preprocessor + classifier).
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from evaluation import METRIC_NAMES, compute_classification_metrics, positive_class_scores
from model_catalog import MODEL_CATALOG
from preprocessing import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_preprocessor,
    purchase_labels,
)

PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "online_shoppers_intention.csv"
TEST_PATH = PROJECT_DIR / "test_data.csv"
MODEL_DIR = PROJECT_DIR / "model"
RANDOM_STATE = 42
TEST_SIZE = 0.2


# Method: load_sessions
# Description: Reads the UCI shopper file, keeps the 17 features and Revenue, and encodes the target as 0 or 1.
# Params: path (Path) - path to online_shoppers_intention.csv
def load_sessions(path: Path) -> pd.DataFrame:
    sessions = pd.read_csv(path)
    missing = [column for column in FEATURE_COLUMNS + [TARGET_COLUMN] if column not in sessions.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")
    sessions = sessions[FEATURE_COLUMNS + [TARGET_COLUMN]].dropna()
    sessions[TARGET_COLUMN] = purchase_labels(sessions[TARGET_COLUMN])
    return sessions


# Method: fit_catalog_model
# Description: Fits one catalog classifier with a small grid search on the training split only.
# Params: spec (dict) - model name, estimator, parameter grid, and scaling flag
# features (DataFrame) - training session columns
# labels (Series) - training purchase labels
def fit_catalog_model(spec, features, labels):
    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(spec["scale_numeric"])),
            ("classifier", spec["estimator"]()),
        ]
    )
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=spec["param_grid"],
        scoring="f1",
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=-1,
        refit=True,
    )
    search.fit(features, labels)
    return search


# Method: metric_row
# Description: Formats one model's held-out scores as a Markdown table row.
# Params: name (str) - display name of the classifier
# metrics (dict) - Accuracy, AUC, Precision, Recall, F1, and MCC
def metric_row(name, metrics):
    cells = [name] + [f"{metrics[metric]:.4f}" for metric in METRIC_NAMES]
    return "| " + " | ".join(cells) + " |"


# Method: main
# Description: Splits the dataset, writes test_data.csv, trains the five models, and saves each pipeline.
# Params: none
def main():
    sessions = load_sessions(DATA_PATH)
    labels = sessions[TARGET_COLUMN]
    features = sessions[FEATURE_COLUMNS]
    purchase_rate = float(labels.mean())
    print(f"sessions={len(sessions)} features={len(FEATURE_COLUMNS)} purchase_rate={purchase_rate:.4f}")

    train_features, test_features, train_labels, test_labels = train_test_split(
        features,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )
    test_frame = test_features.copy()
    test_frame[TARGET_COLUMN] = test_labels.to_numpy()
    test_frame.to_csv(TEST_PATH, index=False)
    print(f"wrote {TEST_PATH} rows={len(test_frame)}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    notes = {
        "dataset": "Online Shoppers Purchasing Intention (UCI)",
        "rows": int(len(sessions)),
        "feature_count": len(FEATURE_COLUMNS),
        "purchase_rate": purchase_rate,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "positive_class": "Revenue = 1 (session ended in a purchase)",
        "metric_definitions": {
            "Precision": "purchase class",
            "Recall": "purchase class",
            "F1": "purchase class",
            "AUC": "ROC AUC using the purchase-class score",
            "MCC": "Matthews correlation coefficient",
        },
        "models": [],
    }

    header = "| ML Model Name | " + " | ".join(METRIC_NAMES) + " |"
    divider = "|" + " --- |" * (len(METRIC_NAMES) + 1)
    print(header)
    print(divider)

    for spec in MODEL_CATALOG:
        search = fit_catalog_model(spec, train_features, train_labels)
        fitted = search.best_estimator_
        predictions = fitted.predict(test_features)
        scores = positive_class_scores(fitted, test_features)
        metrics = compute_classification_metrics(test_labels, predictions, scores)
        destination = MODEL_DIR / spec["filename"]
        joblib.dump(fitted, destination, compress=3)
        print(metric_row(spec["name"], metrics))
        print(f"  saved {destination.name} params={search.best_params_} cv_f1={search.best_score_:.4f}")
        notes["models"].append(
            {
                "name": spec["name"],
                "file": spec["filename"],
                "scale_numeric": spec["scale_numeric"],
                "best_params": search.best_params_,
                "cv_f1": float(search.best_score_),
                "test_metrics": metrics,
            }
        )

    notes_path = MODEL_DIR / "training_notes.json"
    notes_path.write_text(json.dumps(notes, indent=2))
    print(f"wrote {notes_path}")


if __name__ == "__main__":
    main()
