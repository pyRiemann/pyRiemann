import numpy as np
from pyriemann.utils.viz import plot_confusion_matrix

def test_confusion_matrix():
    """Test confusion_matrix"""
    target = np.array([0, 1] * 10)
    preds = np.array([0, 1] * 10)
    plot_confusion_matrix(target, preds, ['a', 'b'])
