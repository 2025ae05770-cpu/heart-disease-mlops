"""Domain-driven feature engineering as an sklearn transformer.

`hr_pct_of_max` = thalach / (220 - age): the fraction of the age-predicted maximum
heart rate reached during the exercise test. A low value (chronotropic
incompetence) is a recognised cardiac risk marker and normalises `thalach` for age.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ClinicalFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(list(X.columns), dtype=object)
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.feature_names_in_).copy()
        predicted_max = (220 - X["age"]).clip(lower=1)
        X["hr_pct_of_max"] = X["thalach"] / predicted_max
        return X

    def get_feature_names_out(self, input_features=None):
        return np.append(self.feature_names_in_, "hr_pct_of_max")
