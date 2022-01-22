# -*- coding: utf-8 -*-
#
# @Author   : Yuxiang WU
# @Email    : elephantameler@gmail.com

"""
    Reference: https://scikit-learn.org/stable/modules/model_evaluation.html
"""

import numpy as np
from sklearn.metrics import top_k_accuracy_score
from sklearn.model_selection import GridSearchCV

__all__ = [
    "top_k_accuracy_score",
    "GridSearchCV"
    "top_1_accuracy_score",
    "top_k_accuracy",

]


def top_1_accuracy_score(label: np.ndarray, pred: np.ndarray):
    """ the number of unique label used in top_k_accuracy_score must equal the second shape of pred, but this func doesn't meet the constraint. """
    label = label.squeeze()
    assert pred.shape[0] == label.shape[0], "pred and label art not matching!"
    re = np.sum(np.argmax(pred, axis = 1) == label) / len(label)
    return re


def top_k_accuracy(label: np.ndarray, pred: np.ndarray, k: int = 3):
    label = label.squeeze()
    assert pred.shape[0] == label.shape[0], "pred and label art not matching!"
    re = len(np.where((np.argsort(-pred, axis = 1)[:, :k] - label[:, np.newaxis]) == 0)[0]) / len(label)
    return re
