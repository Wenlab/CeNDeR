# -*- coding: utf-8 -*-
#
# @Author   : Yuxiang WU
# @Email    : elephantameler@gmail.com

import math
import cmath


def make_elliptic_cylinder_shell_density_func(neuronal_ids,
                                              regions,
                                              neurons,
                                              width_scale,
                                              spans,
                                              num_unit,
                                              norm = False):
    """
    using the region
    :param neuronal_ids:
    :param regions:
    :param neurons:
    :param width_scale:
    :param spans:
    :param num_unit:
    :param norm:
    :return:
    """

    features = list()
    width_span, height_span, z_span = spans

    rho_span = math.sqrt(width_span * width_span + height_span * height_span)
    rho_unit_length, z_unit_length = rho_span / num_unit, z_span / num_unit

    for nidx, (nx, ny, nz) in zip(neuronal_ids, neurons):
        feature_vector = [0] * num_unit
        for ridx, res in zip(neuronal_ids, regions):
            if nidx != ridx:
                for rx, ry, rz in res:
                    rho = math.sqrt(pow(width_scale * (rx - nx), 2) + pow(ry - ny, 2))  # elliptic counting
                    z_dis = abs(rz - nz)
                    feature_vector[int(min(max(rho // rho_unit_length, z_dis // z_unit_length), num_unit - 1))] += 1

        feature_vector = feature_vector if not norm else [v / sum(feature_vector) for v in feature_vector]
        features.append(feature_vector)

    return features


def make_blocky_elliptic_cylinder_shell_density_func(neuronal_ids,
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

    :param neuronal_ids:
    :param regions:
    :param neurons:
    :param width_scale:
    :param spans:
    :param num_unit:
    :param norm:
    :return:
    """

    features = list()
    (width_span, height_span, z_span) = spans

    rho_span = math.sqrt(width_span * width_span + height_span * height_span)
    rho_unit_length, z_unit_length = rho_span / num_unit, z_span / num_unit

    for nidx, (n_pt) in zip(neuronal_ids, neurons):
        feature_vector = num_unit * 4 * [0]
        for ridx, res in zip(neuronal_ids, regions):
            if nidx != ridx:
                for r_pt in res:
                    idx = calculate_feature_idx_of_four_blocky_distributions(n_pt, r_pt, rho_unit_length, z_unit_length, width_scale, num_unit)
                    feature_vector[idx] += 1

        feature_vector = feature_vector if not norm else [v / sum(feature_vector) for v in feature_vector]
        features.append(feature_vector)

    return features


def calculate_feature_idx_of_four_blocky_distributions(n_pt, r_pt, rho_unit_length, z_unit_length, width_scale, num_unit):
    (nx, ny, nz), (rx, ry, rz) = n_pt, r_pt
    # part index
    phi = cmath.polar(complex(rx - nx, ry - ny))[1]
    phi_idx = 0 if phi <= 0 else 1
    z_dis = rz - nz
    z_idx = 0 if z_dis <= 0 else 1
    rho = math.sqrt(pow(width_scale * (rx - nx), 2) + pow(ry - ny, 2))  # elliptic counting
    idx = (z_idx * 2 + phi_idx) * num_unit + int(min(max(rho // rho_unit_length, abs(z_dis) // z_unit_length), num_unit - 1))

    return idx
