# -*- coding: utf-8 -*-
#
# @Author   : Yuxiang WU
# @Email    : elephantameler@gmail.com

import math
import cmath
import numpy as np
from itertools import permutations


def make_blocky_elliptic_cylinder_shell_density_func4region(neuronal_ids,
                                                            regions,
                                                            neurons,
                                                            width_scale,
                                                            spans,
                                                            num_unit,
                                                            norm = False):
    """
        This version consider relative solid C.elegans' posture: (left-right, head-body) - (z, phi).
            We divide it into 4 blocks to build distributions (z, phi): (0,0 | -,-) (0,1 | -,+) (1,0 | +.-) (1,1 | +,+)

        ATTENTION: using the region to count
        Visualization code: https://github.com/Wenlab/C.elegans-Neuron-Recognition-Algorithm/issues/8#issuecomment-947274645
    """

    features = np.zeros((len(neurons), 4 * num_unit), dtype = np.float32)
    (width_span, height_span, z_span) = spans

    rho_span = math.sqrt(width_span * width_span + height_span * height_span)
    rho_unit_length, z_unit_length = rho_span / num_unit, z_span / num_unit

    for j, (nidx, n_pt) in enumerate(zip(neuronal_ids, neurons)):
        for ridx, res in zip(neuronal_ids, regions):
            if nidx != ridx:
                for r_pt in res:
                    idx = calculate_feature_idx_of_four_blocky_distributions(n_pt, r_pt, rho_unit_length, z_unit_length, width_scale, num_unit)
                    features[j][idx] += 1.0
    features = features if not norm else features / np.sum(features, axis = 1, keepdims = True)
    return features


def make_blocky_elliptic_cylinder_shell_density_func_4object(neurons,
                                                             width_scale,
                                                             spans,
                                                             num_unit,
                                                             norm = False):
    """
        This version consider relative solid C.elegans' posture: (left-right, head-body) - (z, phi).
            We divide it into 4 blocks to build distributions (z, phi): (0,0 | -,-) (0,1 | -,+) (1,0 | +.-) (1,1 | +,+)

        ATTENTION: using the region to count
        Visualization code: https://github.com/Wenlab/C.elegans-Neuron-Recognition-Algorithm/issues/8#issuecomment-947274645
    """

    features = np.zeros((len(neurons), 4 * num_unit), dtype = np.float32)
    (width_span, height_span, z_span) = spans

    rho_span = math.sqrt(width_span * width_span + height_span * height_span)
    rho_unit_length, z_unit_length = rho_span / num_unit, z_span / num_unit

    for nidx1, (n_pt1) in enumerate(neurons):
        for nidx2, n_pt2 in enumerate(neurons):
            if nidx1 != nidx2:
                idx = calculate_feature_idx_of_four_blocky_distributions(n_pt1, n_pt2, rho_unit_length, z_unit_length, width_scale, num_unit)
                features[nidx1][idx] += 1.0
    features = features if not norm else features / np.sum(features, axis = 1, keepdims = True)

    return features


def calculate_feature_idx_of_four_blocky_distributions(pt1, pt2, rho_unit_length, z_unit_length, width_scale, num_unit):
    (x1, y1, z1), (x2, y2, z2) = pt1, pt2
    # part index
    phi_idx = 0 if cmath.polar(complex(x2 - x1, y2 - y1))[1] <= 0 else 1
    z_dis = z2 - z1
    z_idx = 0 if z_dis <= 0 else 1
    rho = math.sqrt(pow(width_scale * (x2 - x1), 2) + pow(y2 - y1, 2))  # elliptic counting
    idx = (z_idx * 2 + phi_idx) * num_unit + int(min(max(rho // rho_unit_length, abs(z_dis) // z_unit_length), num_unit - 1))

    return idx
