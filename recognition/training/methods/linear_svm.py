# -*- coding: utf-8 -*-
#
# @Author   : Yuxiang WU
# @Email    : elephantameler@gmail.com

from typing import Dict

import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC

from common_utils.prints import print_log_message
from recognition.training.extract_training_data import neurons2data


# warnings.filterwarnings("ignore")

def linear_svm_run(neurons: Dict
                   ):
    """
    https://www.cnblogs.com/solong1989/p/9620170.html

    :param neurons:
    :return:
    """
    results = list()

    param_grid = {
        # "penalty": ['l1', 'l2'],
        # "loss": ['hinge', 'squared_hinge'],
        # "dual": [True, False],
        "C": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11,
              0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19
              ] + [
                 5., 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.,
                 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 7., 7.1,
                 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 8., 8.1, 8.2,
                 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 9., 9.1, 9.2, 9.3,
                 9.4, 9.5, 9.6, 9.7, 9.8, 9.9, 10., 10.1, 10.2, 10.3, 10.4,
                 10.5, 10.6, 10.7, 10.8, 10.9, 11., 11.1, 11.2, 11.3, 11.4, 11.5,
                 11.6, 11.7, 11.8, 11.9, 12., 12.1, 12.2, 12.3, 12.4, 12.5, 12.6,
                 12.7, 12.8, 12.9, 13., 13.1, 13.2, 13.3, 13.4, 13.5, 13.6, 13.7,
                 13.8, 13.9, 14., 14.1, 14.2, 14.3, 14.4, 14.5, 14.6, 14.7, 14.8,
                 14.9, 15., 15.1, 15.2, 15.3, 15.4, 15.5, 15.6, 15.7, 15.8, 15.9,
                 16., 16.1, 16.2, 16.3, 16.4, 16.5, 16.6, 16.7, 16.8, 16.9, 17.,
                 17.1, 17.2, 17.3, 17.4, 17.5, 17.6, 17.7, 17.8, 17.9, 18., 18.1,
                 18.2, 18.3, 18.4, 18.5, 18.6, 18.7, 18.8, 18.9, 19., 19.1, 19.2,
                 19.3, 19.4, 19.5, 19.6, 19.7, 19.8, 19.9, 20., 20.1, 20.2, 20.3,
                 20.4, 20.5, 20.6, 20.7, 20.8, 20.9, 21., 21.1, 21.2, 21.3, 21.4,
                 21.5, 21.6, 21.7, 21.8, 21.9, 22., 22.1, 22.2, 22.3, 22.4, 22.5,
                 22.6, 22.7, 22.8, 22.9, 23., 23.1, 23.2, 23.3, 23.4, 23.5, 23.6,
                 23.7, 23.8, 23.9, 24., 24.1, 24.2, 24.3, 24.4, 24.5, 24.6, 24.7,
                 24.8, 24.9]
    }

    for m in [(0,), (1,), (0, 1)]:
        X, y, num_ids, id_map = neurons2data(neurons.copy(), is_common_id = True, mode = m)
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

        grid_search = GridSearchCV(LinearSVC(max_iter = 100000,
                                             ), param_grid, n_jobs = -1, verbose = 1, cv = 5)
        grid_search.fit(X, y)
        grid_search.fit(X, y)

        print_log_message(
                f"| Mode: {m}\n best score are {grid_search.best_score_}\n best parameters are {grid_search.best_params_} |")
        results.append(pd.DataFrame(grid_search.cv_results_))

    return results
