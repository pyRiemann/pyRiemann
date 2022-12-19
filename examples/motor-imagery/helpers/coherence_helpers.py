"""
=================================
 Functional Connectivity Helpers
=================================

This file contains helper functions for the functional connectivity example
"""
# Authors: Sylvain Chevallier <sylvain.chevallier@universite-paris-saclay.fr>,
#          Marie-Constance Corsi <marie.constance.corsi@gmail.com>
#
# License: BSD (3-clause)


from pyriemann.utils.base import nearest_pos_def

from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import balanced_accuracy_score
from sklearn.ensemble import StackingClassifier


class NearestSPD(TransformerMixin, BaseEstimator):
    """Transform square matrices to nearest SPD matrices"""

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return nearest_pos_def(X)

    def fit_transform(self, X, y=None):
        return self.transform(X)


def get_results(X, y, all_ppl):
    results = []
    le = LabelEncoder()
    y_ = le.fit_transform(y)
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    for idx, (train, test) in enumerate(cv.split(X, y_)):
        for ppn, ppl in all_ppl.items():
            cvclf = clone(ppl)
            cvclf.fit(X[train], y_[train])
            yp = cvclf.predict(X[test])
            acc = balanced_accuracy_score(y_[test], yp)
            res = {
                "score": acc,
                "pipeline": ppn,
                "split": idx,
                "samples": len(y_),
            }
            results.append(res)
            if isinstance(ppl, StackingClassifier):
                for est_n, est_p in cvclf.named_estimators_.items():
                    ype = est_p.predict(X[test])
                    acc = balanced_accuracy_score(y_[test], ype)
                    res = {
                        "score": acc,
                        "pipeline": est_n,
                        "split": idx,
                        "samples": len(y_),
                    }
                    results.append(res)
    return results
