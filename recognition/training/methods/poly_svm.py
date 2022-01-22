# -*- coding: utf-8 -*-
#
# @Author   : Yuxiang WU
# @Email    : elephantameler@gmail.com

from typing import Dict

import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from common_utils.prints import print_log_message
from recognition.training.extract_training_data import neurons2data


# warnings.filterwarnings("ignore")


def poly_svm_run(neurons: Dict
                 ):
    """
    https://www.cnblogs.com/solong1989/p/9620170.html

    :param neurons:
    :return:
    """
    results = list()

    param_grid = {
        "degree": [1, 2, 3, 4, 5, 6],
        "C"     : [0.01, 0.05, 0.1, 0.2, 0.4, 0.5, 0.6, 0.7, 0.9, 1.0, 2.0, 3.0, 4.0, 5.0,
                   7.0, 10.0, 15.0, 20.0]
    }

    for m in [(0,), (1,), (0, 1)]:
        X, y, num_ids, id_map = neurons2data(neurons.copy(), is_common_id = True, mode = m)
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

        grid_search = GridSearchCV(SVC(kernel = 'poly',
                                       gamma = 'auto',
                                       ), param_grid, n_jobs = -1, verbose = 1, cv = 5)
        grid_search.fit(X, y)

        print_log_message(
                f"| Mode: {m}\n best score are {grid_search.best_score_}\n best parameters are {grid_search.best_params_} |")
        results.append(pd.DataFrame(grid_search.cv_results_))

    return results
