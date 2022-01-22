# -*- coding: utf-8 -*-
#
# @Author   : Yuxiang WU
# @Email    : elephantameler@gmail.com

from typing import Dict

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from common_utils.prints import print_log_message
from recognition.training.extract_training_data import neurons2data


def logistic_reg_run(neurons: Dict
                     ):
    """
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression

    :param neurons:
    :return:
    """

    results = list()

    param_grid = {
        "multi_class": ['multinomial', 'ovr'],
        "C"          : [0.01, 0.05, 0.1, 0.2, 0.4, 0.5, 0.6, 0.7, 0.9, 1.0, 2.0, 3.0, 4.0, 5.0,
                        7.0, 10.0, 15.0, 20.0]
    }

    for m in [(0,), (1,), (0, 1)]:
        X, y, num_ids, id_map = neurons2data(neurons.copy(), is_common_id = True, mode = m)
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

        grid_search = GridSearchCV(LogisticRegression(solver = 'sag',
                                                      max_iter = 10000),
                                   param_grid, n_jobs = -1, verbose = 1, cv = 5)
        grid_search.fit(X, y)

        print_log_message(
                f"| Mode: {m}\n best score are {grid_search.best_score_}\n best parameters are {grid_search.best_params_} |")
        results.append(pd.DataFrame(grid_search.cv_results_))

    return results
