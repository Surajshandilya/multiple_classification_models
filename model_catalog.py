"""The five required classifiers, with small grids searched only on the training split."""

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

# scale_numeric is False for the tree, because split thresholds do not depend on
# standardized units, and for Multinomial Naive Bayes, which requires the
# original non-negative counts, durations, and one-hot codes.
MODEL_CATALOG = [
    {
        "name": "Logistic Regression",
        "filename": "logistic_regression.joblib",
        "scale_numeric": True,
        "estimator": lambda: LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=42,
        ),
        "param_grid": {"classifier__C": [0.1, 1.0, 3.0]},
    },
    {
        "name": "Decision Tree",
        "filename": "decision_tree.joblib",
        "scale_numeric": False,
        "estimator": lambda: DecisionTreeClassifier(
            class_weight="balanced",
            random_state=42,
        ),
        "param_grid": {
            "classifier__max_depth": [4, 6, 10],
            "classifier__min_samples_leaf": [10, 30],
        },
    },
    {
        "name": "KNN",
        "filename": "knn.joblib",
        "scale_numeric": True,
        "estimator": lambda: KNeighborsClassifier(),
        "param_grid": {
            "classifier__n_neighbors": [5, 11, 21],
            "classifier__weights": ["uniform", "distance"],
        },
    },
    {
        "name": "Naive Bayes",
        "filename": "naive_bayes.joblib",
        "scale_numeric": False,
        "estimator": lambda: MultinomialNB(),
        "param_grid": {"classifier__alpha": [0.1, 1.0, 5.0]},
    },
    {
        "name": "Support Vector Machine",
        "filename": "support_vector_machine.joblib",
        "scale_numeric": True,
        "estimator": lambda: LinearSVC(
            class_weight="balanced",
            dual=False,
            max_iter=10000,
            random_state=42,
        ),
        "param_grid": {"classifier__C": [0.1, 1.0, 3.0]},
    },
]
