# -*- coding: utf-8 -*-
#
# @Author   : Yuxiang WU
# @Email    : elephantameler@gmail.com

import cv2
import math
import cmath
import numpy as np
from typing import Dict, List, NewType, Tuple

RhoLike = NewType("RhoLike", float)
PhiLike = NewType("PhiLike", float)
ZLike = NewType("ZLike", int)
DistanceLike = NewType("DistanceLike", float)

CartesianCoordLike = List[int]
PolarCoordLike = Tuple[RhoLike, PhiLike]
CylindricalCoordLike = Tuple[RhoLike, PhiLike, ZLike]


# ----- Degree & Radian --------------------
def degree2radian(degree):
    return math.radians(degree)


def radian2degree(radian):
    return math.degrees(radian)


# ----- Cylindrical Coordinate System --------------------
def calculate_2pt_cylindrical_coordinate(std_pt: CartesianCoordLike, pt: CartesianCoordLike) -> CylindricalCoordLike:
    """

    Type of Input is Cartesian coordinate system,
    Output is Cylindrical coordinate system.
    CCS: https://zh.wikipedia.org/wiki/%E5%9C%93%E6%9F%B1%E5%9D%90%E6%A8%99%E7%B3%BB
    :param std_pt: List[x, y, z]
    :param pt: List[x, y, z]
    :return: Tuple[rou, phi, z]
    """
    rel_pt = [p - std_p for std_p, p in zip(std_pt, pt)]
    rho, phi = calc_cylindrical_rho_phi(rel_pt)
    z = calc_cylindrical_z(rel_pt)
    return rho, phi, z


def calc_cylindrical_rho_phi(pt: CartesianCoordLike) -> Tuple[RhoLike, PhiLike]:
    """
    This function uses complex number to implement
    transfer between Cartesian Coordinate System and Polar Coordinate System in 2D.

    phi belongs to (-pi, pi]

    :param pt: List[x, y, z]
    :return: Tuple[rho, phi]
    """
    cartesian = complex(pt[0], pt[1])  # return (real, imaginary)
    rho, phi = cmath.polar(cartesian)  # phi is radian belonging to (-pi, pi]
    return rho, phi


def calc_cylindrical_z(pt: CartesianCoordLike) -> ZLike:
    """

    :param pt: List[x, y, z]
    :return: z
    """
    return pt[2]


# ----- Cartesian Coordinate System --------------------

def calc_cartesian_coordinate(pt: PolarCoordLike, is_degree: bool = True):
    rho, phi = (pt[0], degree2radian(pt[1])) if is_degree else pt
    x, y = cmath.rect(rho, phi).real, cmath.rect(rho, phi).imag
    return x, y


def calc_spherical_distance(pt1: List[float],
                            pt2: List[float]) -> DistanceLike:
    distance: DistanceLike = math.sqrt(sum([pow(p1 - p2, 2) for p1, p2 in zip(pt1, pt2)]))
    return distance


def calc_spherical_dists(std_pt: List[float],
                         other_pts: List[List[float]]
                         ) -> List[DistanceLike]:
    dists = [calc_spherical_distance(std_pt, pt) for pt in other_pts]
    return dists


def calc_spherical_dist_matrix(pts: List[List[float]]):
    dist_matrix = [calc_spherical_dists(pt, pts) for pt in pts]
    return dist_matrix


# ----- Representation of 3D neuron (list of 2d bboxes) --------------------
def calc_mean_pt_of_bbox(bboox: List[float]):
    # xmin, ymin, xmax, ymax, z -> x, y, z
    return [(bboox[0] + bboox[2]) * 0.5, (bboox[1] + bboox[3]) * 0.5, bboox[4]]


def calc_mean_pts_of_3d_neuron(neuron_pts: List[List]):
    """

    :param neuron_pts: a neuron with raw bboxes (xmin, ymin, xmax, ymax, z)
    :return: a neuron with bboxes after mean operation
    """
    neuron_mean_pts = [calc_mean_pt_of_bbox(bbox) for bbox in neuron_pts]
    return neuron_mean_pts


def calc_mean_pt_of_3d_neuron(neuron_mean_pts: List[List]) -> List[float]:
    """

    :param neuron_mean_pts: a neuron with bboxes after mean operation
    :return: [x, y, z]
    """

    mean_x = sum([b[0] for b in neuron_mean_pts]) / len(neuron_mean_pts)
    mean_y = sum([b[1] for b in neuron_mean_pts]) / len(neuron_mean_pts)
    mean_z = sum([b[2] for b in neuron_mean_pts]) / len(neuron_mean_pts)
    neuron_mean_pt = [mean_x, mean_y, mean_z]
    return neuron_mean_pt


def calc_mean_pts(neurons: Dict) -> Tuple[List, List, List]:
    """

    :param neurons: {neuron_id: [[xmin, ymin, xmax, ymax, z]]}
    :return: the same order between ids and pts
    """

    ids, regions_mean, neurons_mean = list(), list(), list()
    for key, value in neurons.items():
        neuron_mean_regions = calc_mean_pts_of_3d_neuron(value)
        ids.append(key)
        regions_mean.append(neuron_mean_regions)
        neurons_mean.append(calc_mean_pt_of_3d_neuron(neuron_mean_regions))

    return ids, regions_mean, neurons_mean


# ----- General --------------------
def count_coord_feature(coords: List[float],
                        bins = 10, range = None, norm = False):
    feas, bins = np.histogram(coords, bins = bins, range = range, density = False)
    feas = feas / np.sum(feas) if norm else feas
    feas, bins = feas.tolist(), bins.tolist()
    return feas, bins


# def scale_z_axis1(neurons: Dict, z_scale: float) -> Dict:
#     """
#
#     :param neurons: [z, xmin, ymin, xmax, ymax, is_key] style
#     :param z_scale: scale factor
#     :return: [xmin, ymin, xmax, ymax, z * scale] style
#     """
#     for key, value in neurons.items():
#         neurons[key] = [[bb[1], bb[2], bb[3], bb[4], bb[0] * z_scale] for bb in value]
#     return neurons


def scale_z_axis(neurons: Dict, z_scale: float) -> Dict:
    """
    Ensure z axis is the same unit with xoy plate.
    :param neurons: [xmin, ymin, xmax, ymax, z] style
    :param z_scale: scale factor
    :return: [xmin, ymin, xmax, ymax, z * scale] style
    """

    for key, value in neurons.items():
        neurons[key] = [[bb[0], bb[1], bb[2], bb[3], bb[4] * z_scale] for bb in value]  # Only a calculation on z value to reduce time consumption.
    return neurons


def calc_object_center(contour):
    moments = cv2.moments(contour)
    center = (int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"]))
    return center


def calc_corner_pts(center, contour_pts):
    calc_given_phi_arg = lambda phi: np.argmin([abs(pt[1] - (divmod(phi + math.pi, math.pi * 2)[1] - math.pi)) for pt in polars])

    polars = [calc_cylindrical_rho_phi([pt[0][0] - center[0], pt[0][1] - center[1]]) for pt in contour_pts]

    farthest_pt_arg = np.argmax([p[0] for p in polars])
    refer_radian = polars[farthest_pt_arg][1]

    y1 = tuple(contour_pts[farthest_pt_arg].flatten())
    y2 = tuple(contour_pts[calc_given_phi_arg(refer_radian - math.pi)].flatten())
    x1 = tuple(contour_pts[calc_given_phi_arg(refer_radian - math.pi / 2)].flatten())
    x2 = tuple(contour_pts[calc_given_phi_arg(refer_radian + math.pi / 2)].flatten())

    return y1, y2, x1, x2


def normalize_neurons_range(neurons, standard_diagonal_line: int or float):
    """

    :param neurons: should be refined.
    :param standard_diagonal_line: pre-defined standard length of diagonal line of xoy plate

    :return: neurons, width_scale, [width_span, height_span, z_span]
            width_scale: The length of width is different with height among all volumes, so scaling width
                         could transfer ellipse shell into circle shell to count conveniently on xoy plate.
    """

    regions = np.array([re for res in neurons.copy() for re in res], dtype = np.float32)  # [x, y, z]
    width, height = np.max(regions[:, 0]) - np.min(regions[:, 0]), np.max(regions[:, 1]) - np.min(regions[:, 1])
    scale = standard_diagonal_line / math.sqrt(width * width + height * height)
    neurons = [[[p[0] * scale, p[1] * scale, p[2] * scale] for p in pp] for pp in neurons]  # for knn feature

    width_scale = height / width
    width_span = width * width_scale * scale
    height_span = height * scale
    z_span = (np.max(regions[:, 2]) - np.min(regions[:, 2])) * scale

    return neurons, width_scale, [width_span, height_span, z_span]
