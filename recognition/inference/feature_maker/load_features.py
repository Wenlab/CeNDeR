# -*- coding: utf-8 -*-
#
# @Author   : Yuxiang WU
# @Email    : elephantameler@gmail.com

import numpy as np
from typing import Dict, List

from .neural_knn import make_knn_func
from .refine_celegans_posture import refine_celegans_posture
from .neural_density import make_elliptic_cylinder_shell_density_func, make_blocky_elliptic_cylinder_shell_density_func
from .utils import calc_mean_pt_of_3d_neuron, calc_mean_pts, calc_spherical_dist_matrix, normalize_neurons_range, scale_z_axis


def make_one_volume_neuronal_features(key, volume_3d_result: Dict, ccords: List, args):
    """
    :param key: used during debugging period
    :param volume_3d_result: pt style: [xmin, ymin, xmax, ymax, z]
    :param ccords: [mass_of_center, anterior, posterior, dorsal, ventral]
    :param args: recognition arguments
    :return: Dict type. id: [knn_feature, density_feature]
    """

    # 0. scale z axis to keep the same unit(pixel).
    neurons = scale_z_axis(volume_3d_result, z_scale = args.rec_z_scale)  # (xmin, ymin, xmax, ymax, z)

    # 1. transfer neurons dict into calculable type (now pts with dict type)
    neuronal_ids, regions_pt, neurons_pt = calc_mean_pts(neurons)

    # np.save(f"/home/customer/elegans-neuron-net/supplementary1/rec/neurons_data/{key}_raw.npy", neurons_pt, allow_pickle=True)
    straighten_regions_pt = refine_celegans_posture(regions_pt, ccords = ccords)
    normalized_straighten_regions_pt, width_scale, spans = normalize_neurons_range(straighten_regions_pt, standard_diagonal_line = args.rec_worm_diagonal_line)
    normalized_straighten_neurons_pt = [calc_mean_pt_of_3d_neuron(bb) for bb in normalized_straighten_regions_pt]

    # np.save(f"/home/customer/elegans-neuron-net/supplementary1/rec/neurons_data/{key}_trans.npy", normalized_straighten_neurons_pt, allow_pickle=True)
    # 2. calculate distance matrix
    dist_matrix = np.array(calc_spherical_dist_matrix(normalized_straighten_neurons_pt), dtype = np.float32)

    # 3. make feature vector
    # 3.1 K-nearest neighbor feature
    knn_features = make_knn_func(neurons_pt = np.array(normalized_straighten_neurons_pt, dtype = np.float32),
                                 dist_matrix = dist_matrix,
                                 k_pts = args.rec_knn_k,
                                 rho_span = args.rec_worm_diagonal_line,
                                 z_span = spans[-1],
                                 norm = True,
                                 )

    # 3.2 Neuronal density feature
    density_features = make_blocky_elliptic_cylinder_shell_density_func(neuronal_ids = neuronal_ids,
                                                                        regions = normalized_straighten_regions_pt,
                                                                        neurons = normalized_straighten_neurons_pt,
                                                                        width_scale = width_scale,
                                                                        spans = spans,
                                                                        num_unit = args.rec_des_len,
                                                                        norm = True,
                                                                        )

    neuronal_features = {i: [kf, df] for i, kf, df in zip(neuronal_ids, knn_features, density_features)}

    return neuronal_features
