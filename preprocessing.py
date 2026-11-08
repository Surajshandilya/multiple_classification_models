"""Preprocessing shared by training and the Streamlit app.

Saved pipelines import this module by name, so keep it importable as
``preprocessing`` from the project root.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Ten continuous session measurements. Counts and durations describe how the
# visitor moved through administrative, informational, and product pages.
NUMERIC_FEATURES = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
]

# Seven context fields stored as codes or labels. They are one-hot encoded
# because the integer codes (browser, region, traffic type) are not magnitudes.
CATEGORICAL_FEATURES = [
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
]

TARGET_COLUMN = "Revenue"
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
PURCHASE_LABEL = 1


class CategoricalAsText(BaseEstimator, TransformerMixin):
    """Turn coded fields into strings so one-hot encoding treats them as categories."""

    # Method: fit
    # Description: Accepts the categorical columns and learns nothing, because the cast has no fitted state.
    # Params: self (CategoricalAsText) - transformer instance
    # features (array-like) - categorical columns from the current batch
    # labels (array-like or None) - unused target required by the sklearn fit API
    def fit(self, features, labels=None):
        return self

    # Method: transform
    # Description: Casts each categorical field to text so one-hot encoding treats codes as categories.
    # Params: self (CategoricalAsText) - transformer instance
    # features (array-like) - categorical columns from the current batch
    def transform(self, features):
        frame = pd.DataFrame(features).copy()
        text = frame.astype("string").fillna("missing")
        text.columns = [str(column) for column in text.columns]
        return text

    # Method: get_feature_names_out
    # Description: Passes input column names through so later one-hot names stay traceable.
    # Params: self (CategoricalAsText) - transformer instance
    # input_features (sequence or None) - names of the categorical columns
    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            raise ValueError("input_features is required")
        return np.asarray(input_features, dtype=object)


# Method: build_preprocessor
# Description: Builds the column transformer that scales numeric session fields and one-hot encodes context fields.
# Params: scale_numeric (bool) - True when the classifier needs standardized numeric columns
def build_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    """Build the column transformer used inside each saved model pipeline.

    Numeric session measurements are standardized only for models that depend
    on feature scale. Categorical codes are always one-hot encoded, and unseen
    categories in an uploaded file are ignored instead of raising.
    """
    categorical_pipe = Pipeline(
        steps=[
            ("as_text", CategoricalAsText()),
            (
                "one_hot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    numeric_step = (
        Pipeline(steps=[("scale", StandardScaler())])
        if scale_numeric
        else "passthrough"
    )
    return ColumnTransformer(
        transformers=[
            ("session_measurements", numeric_step, NUMERIC_FEATURES),
            ("session_context", categorical_pipe, CATEGORICAL_FEATURES),
        ]
    )


# Method: purchase_labels
# Description: Maps the Revenue column to 1 for a purchase and 0 for a session with no purchase.
# Params: series (Series) - raw Revenue values as booleans, numbers, or text
def purchase_labels(series: pd.Series) -> pd.Series:
    """Map the Revenue column to 1 (purchase) and 0 (leave without buying)."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype(int)
    if pd.api.types.is_numeric_dtype(series):
        return (series.astype(float) > 0).astype(int)
    text = series.astype(str).str.strip().str.lower()
    purchased = text.isin({"1", "1.0", "true", "yes", "purchase"})
    return purchased.astype(int)
