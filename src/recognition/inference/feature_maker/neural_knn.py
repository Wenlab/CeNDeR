# -*- coding: utf-8 -*-
#
# @Author   : Yuxiang WU
# @Email    : elephantameler@gmail.com

import math
import numpy as np

from .utils import calculate_2pt_cylindrical_coordinate, calc_spherical_dists


def make_knn_func4object(neurons_pt, dist_matrix: np.ndarray, k_pts, rho_span, z_span, norm: bool) -> np.ndarray:
    """For neuron object"""

    neurons_ccoords = np.zeros((len(neurons_pt), 3 * k_pts), dtype = np.float16)
    for n_idx, (std_pt, dists) in enumerate(zip(neurons_pt, dist_matrix)):
        idxes = np.argsort(dists)[1:k_pts + 1]
        feature_vector = np.array([calculate_2pt_cylindrical_coordinate(std_pt, pt) for pt in neurons_pt[idxes]], dtype = np.float16)

        if norm:
            feature_vector[:, 0] = feature_vector[:, 0] / (rho_span * 0.5) - 1  # [-1, 1]
            feature_vector[:, 1] = feature_vector[:, 1] / math.pi  # (-1, 1]
            feature_vector[:, 2] = feature_vector[:, 2] / z_span  # [-1 ,1]
        # feature_vector = np.stack(feature_vector, axis = 1).ravel().tolist() + [0.0] * 3 * (k_pts - len(idxes))  # padding to the length of k_pts * 3. a negative value is processed as zero
        feature_vector = np.stack(feature_vector, axis = 1).ravel()
        neurons_ccoords[n_idx, :len(feature_vector)] = feature_vector  # [Rho, PHi, Z]

    return neurons_ccoords


def make_knn_func4region(neurons_pt, regions_pt: np.ndarray, dist_matrix: np.ndarray, k_pts, rho_span, z_span, norm: bool) -> np.ndarray:
    """For neuron region"""

    neurons_ccoords = np.zeros((len(neurons_pt), 3 * k_pts), dtype = np.float16)
    for n_idx, (std_pt, dists) in enumerate(zip(neurons_pt, dist_matrix)):
        temp = list()
        for idx in np.argsort(dists)[1:]:
            temp.extend(regions_pt[idx])
            if len(temp) >= k_pts:
                break
        region_idxes = np.argsort(calc_spherical_dists(std_pt, temp))[:k_pts]
        feature_vector = np.array([calculate_2pt_cylindrical_coordinate(std_pt, temp[idx]) for idx in region_idxes], dtype = np.float16)
        if norm:
            feature_vector[:, 0] = feature_vector[:, 0] / (rho_span * 0.5) - 1  # [-1, 1]
            feature_vector[:, 1] = feature_vector[:, 1] / math.pi  # (-1, 1]
            feature_vector[:, 2] = feature_vector[:, 2] / z_span  # [-1 ,1]
        # feature_vector = np.stack(feature_vector, axis = 1).ravel().tolist() + [0.0] * 3 * (k_pts - len(idxes))  # padding to the length of k_pts * 3. a negative value is processed as zero
        feature_vector = np.stack(feature_vector, axis = 1).ravel()
        neurons_ccoords[n_idx, :len(feature_vector)] = feature_vector  # [Rho, PHi, Z]

    return neurons_ccoords
