# -*- coding: utf-8 -*-
#
# @Author   : Yuxiang WU
# @Email    : elephantameler@gmail.com

import math
import numpy as np
from typing import List

from .utils import calculate_2pt_cylindrical_coordinate


def make_knn_func(neurons_pt,
                  dist_matrix: np.ndarray,
                  k_pts,
                  norm: bool,
                  rho_span,
                  z_span
                  ) -> List[List]:
    """
    Using neurons

    :param neurons_pt:
    :param dist_matrix: np.ndarray type with n x n size
    :param k_pts: the number of counting neurons, int
    :param norm:
    :param rho_span:
    :param z_span:
    :return: cylindrical coordinate tuple
    """

    neurons_ccoords = list()
    for std_pt, dists in zip(neurons_pt, dist_matrix):
        idxes = np.argsort(dists)[1:k_pts + 1]
        feature_vector = np.array([calculate_2pt_cylindrical_coordinate(std_pt, pt) for pt in neurons_pt[idxes]], dtype = np.float32)

        if norm:
            feature_vector[:, 0] = feature_vector[:, 0] / (rho_span * 0.5) - 1  # [-1, 1]
            feature_vector[:, 1] = feature_vector[:, 1] / math.pi  # (-1, 1]
            feature_vector[:, 2] = feature_vector[:, 2] / z_span  # [-1 ,1]
        feature_vector = np.stack(feature_vector, axis = 1).ravel().tolist() + [0.0] * 3 * (k_pts - len(idxes))  # padding to the length of k_pts * 3. a negative value is processed as zero

        neurons_ccoords.append(feature_vector)  # [Rho, PHi, Z]

    return neurons_ccoords
