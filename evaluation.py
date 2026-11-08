"""Score a fitted pipeline on held-out sessions. This module never refits a model."""

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

METRIC_NAMES = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


# Method: positive_class_scores
# Description: Returns the purchase-class score used only to compute ROC AUC.
# Params: estimator (Pipeline) - fitted model with predict_proba or decision_function
# features (DataFrame) - original session columns
def positive_class_scores(estimator, features):
    """Return a score for the purchase class, used only for AUC.

    Probability of class 1 is preferred. Linear SVM pipelines expose a signed
    decision value instead; the sign is flipped when class 1 is not classes_[1].
    """
    classes = [int(label) for label in list(estimator.classes_)]
    if PURCHASE_CLASS not in classes:
        raise ValueError(f"Expected purchase label {PURCHASE_CLASS} in {classes}")

    if hasattr(estimator, "predict_proba"):
        probability = estimator.predict_proba(features)
        return probability[:, classes.index(PURCHASE_CLASS)]

    scores = estimator.decision_function(features)
    if classes[1] == PURCHASE_CLASS:
        return scores
    return -scores


PURCHASE_CLASS = 1


# Method: compute_classification_metrics
# Description: Computes Accuracy, AUC, purchase Precision, purchase Recall, purchase F1, and MCC.
# Params: y_true (array-like) - actual labels, 1 for a purchase
# y_pred (array-like) - predicted labels
# y_score (array-like) - purchase-class score for each session
def compute_classification_metrics(y_true, y_pred, y_score):
    """Precision, recall, and F1 are for the purchase class (label 1)."""
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "AUC": float(roc_auc_score(y_true, y_score)),
        "Precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
    }


# Method: confusion_counts
# Description: Builds the confusion matrix with rows and columns ordered as no-purchase then purchase.
# Params: y_true (array-like) - actual labels
# y_pred (array-like) - predicted labels
def confusion_counts(y_true, y_pred):
    return confusion_matrix(y_true, y_pred, labels=[0, 1])


# Method: purchase_classification_report
# Description: Builds a classification report for the no-purchase and purchase classes.
# Params: y_true (array-like) - actual labels
# y_pred (array-like) - predicted labels
def purchase_classification_report(y_true, y_pred):
    return classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["No purchase", "Purchase"],
        zero_division=0,
        output_dict=True,
    )
