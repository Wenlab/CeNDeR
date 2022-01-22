# -*- coding: utf-8 -*-
#
# @Author   : Yuxiang WU
# @Email    : elephantameler@gmail.com

from typing import Dict

import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV

from common_utils.prints import print_log_message
from ..extract_training_data import neurons2data


def adaboost_run(neurons: Dict
                 ):
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html

    :param neurons:
    :return:
    """
    results = list()

    param_grid = {"algorithm"    : ["SAMME.R", "SAMME"],
                  "n_estimators" : [150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290],
                  "learning_rate": [0.001, 0.005, 0.0075, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18,
                                    0.2, 0.5, 1]}
    # X, y = neurons2data(neurons.copy(), is_common_id=True, mode=(0, 1))
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    # model = AdaBoostClassifier(n_estimators=220,
    #                            learning_rate=0.12,
    #                            ).fit(X_train, y_train)
    # y_pred = model.predict(X_test)
    # print(f"Accuracy:{metrics.accuracy_score(y_test, y_pred)}")

    for m in [(0,), (1,)]:
        # nan situation will occur when m = (0, 1)
        X, y, num_ids, id_map = neurons2data(neurons.copy(), is_common_id = True, mode = m)
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

        grid_search = GridSearchCV(AdaBoostClassifier(), param_grid, n_jobs = -1, verbose = 1, cv = 5)
        grid_search.fit(X, y)

        print_log_message(
                f"| Mode: {m}\n best score are {grid_search.best_score_}\n best parameters are {grid_search.best_params_} |")
        results.append(pd.DataFrame(grid_search.cv_results_))

    return results
