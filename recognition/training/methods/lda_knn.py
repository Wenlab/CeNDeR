# -*- coding: utf-8 -*-
#
# @Author   : Yuxiang WU
# @Email    : elephantameler@gmail.com

from typing import Dict

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from common_utils.prints import print_log_message
from recognition.training.extract_training_data import neurons2data


def lda_knn_run(neurons: Dict
                ):
    results = list()

    param_grid = {

    }

    for m in [(0,), (1,), (0, 1)]:
        for n_com in range(1, 40):
            X, y, num_ids, id_map = neurons2data(neurons.copy(), is_common_id = True, mode = m)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)

            # --------------------  How to use multi-estimator in grid search --------------------
            pca = PCA(n_components = n_com)
            pca.fit(X_train, y_train)
            X_new = pca.transform(X)

            grid_search = GridSearchCV(KNeighborsClassifier(),
                                       param_grid, n_jobs = -1, verbose = 1, cv = 5)
            grid_search.fit(X_new, y)

            print_log_message(
                    f"| Mode: {m}\n n_components: {n_com}, \n best score are {grid_search.best_score_}\n best parameters are {grid_search.best_params_} |")
            results.append(pd.DataFrame(grid_search.cv_results_))

        return results
