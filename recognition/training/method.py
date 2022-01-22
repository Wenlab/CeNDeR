# -*- coding: utf-8 -*-
#
# @Author   : Yuxiang WU
# @Email    : elephantameler@gmail.com

from common_utils.prints import print_error_message, print_log_message


def training_procedure(method: str, neuronal_feas, **kwargs):
    """
        classification for ML: https://www.cnblogs.com/qiuyuyu/p/11399697.html
        sklearn: https://scikit-learn.org/stable/supervised_learning.html#supervised-learning
        chinese ver: https://www.cntofu.com/book/170/docs/1.md
    :param method: method name
    :param neuronal_feas: feature vector of neurons
    :param args:
    :param kwargs:
    :return:
    """

    print_log_message(f"{method} is running -----------------  ")

    if method.lower() == "adaboost":
        from recognition.training.methods.adaboost import adaboost_run

        results = adaboost_run(neuronal_feas)

    elif method.lower() == "linear_svm":
        from recognition.training.methods.linear_svm import linear_svm_run

        results = linear_svm_run(neuronal_feas)

    elif method.lower() == "rbf_svm":
        from recognition.training.methods.rbf_svm import rbf_svm_run

        results = rbf_svm_run(neuronal_feas)

    elif method.lower() == "poly_svm":
        from recognition.training.methods.poly_svm import poly_svm_run

        results = poly_svm_run(neuronal_feas)

    elif method.lower() == "linear_reg":
        from recognition.training.methods.logistic_reg import logistic_reg_run

        results = logistic_reg_run(neuronal_feas)

    elif method.lower() == 'deep_learning':
        from recognition.training.methods.deep_learning import deep_learning_run

        results = deep_learning_run(neuronal_feas, **kwargs)

    elif method.lower() == 'pca_knn':
        from recognition.training.methods.pca_knn import pca_knn_run

        results = pca_knn_run(neuronal_feas)

    elif method.lower() == 'lda_knn':
        from recognition.training.methods.lda_knn import lda_knn_run

        results = lda_knn_run(neuronal_feas)

    else:
        print_error_message(f"{method} method doesn't exist! Please check your method name!")

    return results
