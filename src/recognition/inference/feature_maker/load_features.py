# -*- coding: utf-8 -*-
#
# @Author   : Yuxiang WU
# @Email    : elephantameler@gmail.com

import numpy as np
from typing import Dict, List

from .neural_knn import make_knn_func4region, make_knn_func4object
from .neural_density import make_blocky_elliptic_cylinder_shell_density_func4region, make_blocky_elliptic_cylinder_shell_density_func_4object
from .refine_celegans_posture import refine_celegans_posture, refine_celegans_posture_4ptcloud
from .utils import calc_mean_pt_of_3d_neuron, calc_mean_pts, calc_spherical_dist_matrix, normalize_neurons_range, scale_z_axis, normalize_neurons_range_4ptcloud


def make_one_volume_neuronal_features(key, volume_3d_result: Dict, ccords: List, mode: int = 0, **kwargs):
    """
    0: density uses region, knn object;
    1: density uses object, knn object;
    2: density uses region, knn region;
    3: density uses object, knn region.
    :param key: used during debugging period
    :param volume_3d_result: pt style: [xmin, ymin, xmax, ymax, z]
    :param ccords: [mass_of_center, anterior, posterior, dorsal, ventral]
    :param mode: type of two feature vectors.
    :param kwargs: recognition arguments
    :return: Dict type. id: [knn_feature, density_feature]
    """
    if kwargs.get("args"):
        rec_z_scale = kwargs["args"].rec_z_scale
        rec_worm_diagonal_line = kwargs["args"].rec_worm_diagonal_line
        rec_knn_k = kwargs["args"].rec_knn_k
        rec_des_len = kwargs["args"].rec_des_len

    else:
        rec_z_scale = kwargs["rec_z_scale"]
        rec_worm_diagonal_line = kwargs["rec_worm_diagonal_line"]
        rec_knn_k = kwargs["rec_knn_k"]
        rec_des_len = kwargs["rec_des_len"]

    assert mode in (0, 1, 2, 3)

    # 0. scale z axis to keep the same unit(pixel).
    neurons = scale_z_axis(volume_3d_result, z_scale = rec_z_scale)  # (xmin, ymin, xmax, ymax, z)

    # 1. transfer neurons dict into calculable type (now pts with dict type)
    neuronal_ids, regions_pt, neurons_pt = calc_mean_pts(neurons)
    # np.save(f"/home/cbmi/elegans-neuron-net/supplementary1/rec/neurons_data/{key}_raw.npy", neurons_pt, allow_pickle=True)

    straighten_regions_pt = refine_celegans_posture(regions_pt, ccords = ccords, jitter_scope = rec_worm_diagonal_line * 0.025)
    normalized_straighten_regions_pt, width_scale, spans = normalize_neurons_range(straighten_regions_pt, standard_diagonal_line = rec_worm_diagonal_line)
    normalized_straighten_neurons_pt = [calc_mean_pt_of_3d_neuron(bb) for bb in normalized_straighten_regions_pt]
    # np.save(f"/home/cbmi/elegans-neuron-net/supplementary1/rec/neurons_data/{key}_trans.npy", normalized_straighten_neurons_pt, allow_pickle=True)

    # 2. calculate distance matrix
    dist_matrix = np.array(calc_spherical_dist_matrix(normalized_straighten_neurons_pt), dtype = np.float32)

    # 3. make feature vector
    # 3.1 K-nearest neighbor feature
    if (mode == 0) or (mode == 1):
        knn_features = make_knn_func4object(np.array(normalized_straighten_neurons_pt, dtype = np.float32), dist_matrix, rec_knn_k, rec_worm_diagonal_line, spans[-1], norm = True)
    else:
        knn_features = make_knn_func4region(np.array(normalized_straighten_neurons_pt, dtype = np.float32), normalized_straighten_regions_pt, dist_matrix,
                                            rec_knn_k, rec_worm_diagonal_line, spans[-1], norm = True)

    # 3.2 Neuronal density feature
    if (mode == 0) or (mode == 2):
        density_features = make_blocky_elliptic_cylinder_shell_density_func4region(neuronal_ids, normalized_straighten_regions_pt,
                                                                                   normalized_straighten_neurons_pt, width_scale, spans, rec_des_len, norm = True)
    else:
        density_features = make_blocky_elliptic_cylinder_shell_density_func_4object(normalized_straighten_neurons_pt, width_scale, spans, rec_des_len, norm = True)

    neuronal_features = np.concatenate((knn_features, density_features), axis = 1)

    return np.array(neuronal_ids, dtype = np.int), normalized_straighten_neurons_pt, neuronal_features


def make_one_volume_neuronal_features_4ptcloud(neurons_pt: List, ccords: List, **kwargs):
    """
    :param key: used during debugging period
    :param neurons_pt: pt style: [xmin, ymin, xmax, ymax, z]
    :param neuronal_ids:
    :param ccords: [mass_of_center, anterior, posterior, dorsal, ventral]
    :param kwargs: recognition arguments
    :return: Dict type. id: [knn_feature, density_feature]
    """
    if kwargs.get("args"):
        rec_z_scale = kwargs["args"].rec_z_scale
        rec_worm_diagonal_line = kwargs["args"].rec_worm_diagonal_line
        rec_knn_k = kwargs["args"].rec_knn_k
        rec_des_len = kwargs["args"].rec_des_len
    else:
        rec_z_scale = kwargs["rec_z_scale"]
        rec_worm_diagonal_line = kwargs["rec_worm_diagonal_line"]
        rec_knn_k = kwargs["rec_knn_k"]
        rec_des_len = kwargs["rec_des_len"]

    # neurons_pt[:, 2] = neurons_pt[:, 2] * rec_z_scale
    if ccords is not None:
        neurons_pt = refine_celegans_posture_4ptcloud(neurons_pt, ccords = ccords, jitter_scope = rec_worm_diagonal_line * 0.025)
    normalized_straighten_neurons_pt, width_scale, spans = normalize_neurons_range_4ptcloud(np.array(neurons_pt), standard_diagonal_line = rec_worm_diagonal_line)

    # 2. calculate distance matrix
    dist_matrix = np.array(calc_spherical_dist_matrix(normalized_straighten_neurons_pt), dtype = np.float16)

    # 3. make feature vector
    # 3.1 K-nearest neighbor feature
    knn_features = make_knn_func4object(neurons_pt = np.array(normalized_straighten_neurons_pt, dtype = np.float32),
                                        dist_matrix = dist_matrix,
                                        k_pts = rec_knn_k,
                                        rho_span = rec_worm_diagonal_line,
                                        z_span = spans[-1],
                                        norm = True)

    # 3.2 Neuronal density feature
    density_features = make_blocky_elliptic_cylinder_shell_density_func_4object(neurons = normalized_straighten_neurons_pt,
                                                                                width_scale = width_scale,
                                                                                spans = spans,
                                                                                num_unit = rec_des_len,
                                                                                norm = True)

    neuronal_features = np.concatenate((knn_features, density_features), axis = 1)

    return normalized_straighten_neurons_pt, neuronal_features
