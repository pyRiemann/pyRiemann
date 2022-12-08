"""
=================================
 Functional Connectivity with MNE
=================================

This module is designed to compute functional connectivity metrics on
MOABB datasets.
"""
# Authors: Sylvain Chevallier <sylvain.chevallier@universite-paris-saclay.fr>,
#          Marie-Constance Corsi <marie.constance.corsi@gmail.com>
#
# License: BSD (3-clause)


import numpy as np

from pyriemann.utils.test import is_pos_def

from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import balanced_accuracy_score
from sklearn.ensemble import StackingClassifier


def nearestPD(A, reg=1e-6):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): htttps://doi.org/10.1016/0024-3795(88)90223-6
    """
    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if is_pos_def(A3):
        # Regularize if already PD
        ei, ev = np.linalg.eigh(A3)
        if np.min(ei) / np.max(ei) < reg:
            A3 = ev @ np.diag(ei + reg) @ ev.T
        return A3

    spacing = np.spacing(np.linalg.norm(A))
    I = np.eye(A.shape[0])  # noqa
    k = 1
    while not is_pos_def(A3, fast_mode=False):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += I * (-mineig * k ** 2 + spacing)
        k += 1

    # Regularize
    ei, ev = np.linalg.eigh(A3)
    if np.min(ei) / np.max(ei) < reg:
        A3 = ev @ np.diag(ei + reg) @ ev.T
    return A3


class EnsureSPD(TransformerMixin, BaseEstimator):
    """Getting connectivity features from mat files"""

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        Xspd = np.empty_like(X)
        for i, mat in enumerate(X):
            Xspd[i, :, :] = nearestPD(mat)
        return Xspd

    def fit_transform(self, X, y=None):
        transf = self.transform(X)
        return transf


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
