# -*- coding: utf-8 -*-
#
# @Author   : Yuxiang WU
# @Email    : elephantameler@gmail.com

import os
import sys
import cv2
import json
import math
import numpy as np
from scipy.io import loadmat
from typing import List, Tuple
import matplotlib.pyplot as plt

sys.path.append("../../../")

from src.common_utils.prints import pad_num, print_warning_message
from src.common_utils.dataset_building import get_stack_name
from src.preprocessing.inference.anno2json import save_anno_as_json
from src.recognition.inference.feature_maker.utils import calc_cylindrical_rho_phi, calc_mean_pt_of_3d_neuron, calc_mean_pt_of_bbox

plt.rcParams['figure.figsize'] = (10, 10)
CartesianCoordLike = Tuple[int, int]


# ----- General --------------------
def load_worm_mat(path: str, is_normalize: bool = False, z_range = (0, 18), pixel_range = (0, 1600)):
    stack = [np.stack(v[:, 0], axis = 0) for k, v in loadmat(path).items() if k.lower().startswith('img') or k.lower().startswith('image')][0][..., z_range[0]:z_range[1]]
    mins = np.min(stack, axis = (1, 2, 3), keepdims = True)
    maxs = np.max(stack, axis = (1, 2, 3), keepdims = True)
    mins_flatten = mins.squeeze()
    maxs_flatten = maxs.squeeze()
    min_ids = np.atleast_1d(np.argwhere(mins_flatten < pixel_range[0]).squeeze())
    max_ids = np.atleast_1d(np.argwhere(maxs_flatten > pixel_range[1]).squeeze())
    for i in min_ids:
        print_warning_message(f"the minimum of Volume {i + 1} on {path} is lower than {pixel_range[0]}, which is {mins_flatten[i]}")
        np.clip(stack[i], a_min = pixel_range[0], a_max = pixel_range[1], out = stack[i])
    for i in max_ids:
        print_warning_message(f"the maximum of Volume {i + 1} on {path} is bigger than {pixel_range[1]}, which is {maxs_flatten[i]}")
        np.clip(stack[i], a_min = pixel_range[0], a_max = pixel_range[1], out = stack[i])
    if is_normalize:
        mins = np.where(mins < pixel_range[0], pixel_range[0], mins)
        maxs = np.where(maxs > pixel_range[1], pixel_range[1], maxs)
        stack = (stack - mins) / (maxs - mins)
    return stack


def normalize_volume(volume: np.ndarray):
    return (volume - volume.min()) / (volume.max() - volume.min())


def compress_contour(ctr, ratio = 0.005):
    ctr = [cv2.approxPolyDP(ctr, ratio * cv2.arcLength(ctr, True), True)]
    return ctr


def get_image4processing(volume, is_max = True):
    """
    the projection of maximum value along z-axis is better than mean value.
    ATTENTION: Volume from WenLab system need to eliminate the last two slice.
    :param volume: H x W x S, (S = 22 or 18 for 23 or 20) in WenLab system
    :param is_max:
    :return:
    """

    image = np.max(volume, axis = -1) if is_max else np.mean(volume, axis = -1)

    return image


# ----- Region finding --------------------

def get_head_contour(std_img, warning_head_area = 25000.0, is_convex: bool = False):
    """
    More rigorous condition to tightly crop head region
    :param is_convex:
    :param std_img:
    :return:
    """

    # median blur for pepper & salt noise
    image = cv2.medianBlur(std_img, 5)

    # close operation
    image = cv2.dilate(image, np.ones((13, 13), np.uint8), iterations = 3)
    image = cv2.erode(image, np.ones((9, 9), np.uint8), iterations = 3)

    binary_img = (image > (image.mean() + image.std())).astype(np.uint8)
    ctrs, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(ctrs) == 0:
        raise ValueError("Can't find any area in the image")
    head_ctr = ctrs[np.argmax([cv2.contourArea(ctr) for ctr in ctrs])]

    # if area is bigger than warning_head_area, the area may be the union of head and tail
    is_head_region_warning = (cv2.contourArea(head_ctr) > warning_head_area)

    head_ctr = [cv2.convexHull(head_ctr)] if is_convex else head_ctr

    return head_ctr, is_head_region_warning


def draw_head_region(std_img, head_ctr):
    heatmap = np.zeros(std_img.shape, np.uint8)
    cv2.fillPoly(heatmap, head_ctr, (1,), cv2.LINE_AA)
    return heatmap


def get_tail_contour(std_img, head_heatmap, noise_area = 500.0, is_convex: bool = False):
    """
        More loose condition to keep neuron region as much as possible
        :param std_img:
        :return:
        """
    # median blur for pepper & salt noise
    image = cv2.medianBlur(std_img, 5)

    image = cv2.dilate(image, np.ones((13, 13), np.uint8), iterations = 3)
    image = cv2.erode(image, np.ones((9, 9), np.uint8), iterations = 3)

    binary_img = (image > image.mean()) * (1 - head_heatmap)

    ctrs, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(ctrs) == 0:
        raise ValueError("Can't find any area in the image")
    # bboxes = [cv2.boundingRect(ctr) for ctr in ctrs]
    areas = [cv2.contourArea(ctr) for ctr in ctrs]

    tail_args = [i for i in range(len(ctrs)) if (areas[i] > noise_area)]
    tail_ctrs = [ctrs.pop(i) for i in tail_args[::-1]]

    noise_ctrs = ctrs

    if is_convex:
        tail_ctrs = [[cv2.convexHull(ctr)] for ctr in tail_ctrs]
        noise_ctrs = [[cv2.convexHull(ctr)] for ctr in noise_ctrs]

    return tail_ctrs, noise_ctrs


def crop_worm_region(volume):
    """
    the projection of maximum value along z-axis is better than mean value.
    :param volume: H x W x S
    :return:
    """
    max_projection = np.max(volume, axis = -1)
    blur_image = cv2.blur(max_projection, (13, 13))
    image = cv2.dilate(blur_image, np.ones((11, 11), np.uint8), iterations = 4)
    image = cv2.erode(image, np.ones((9, 9), np.uint8), iterations = 3)

    tail_image_candidate = image > image.mean()
    head_image_candidate = image > (image.mean() + image.std())

    return max_projection, tail_image_candidate, head_image_candidate


# ----- Building C.elegans Coordinate system --------------------
def calc_object_center(contour):
    """

    :param contour: if contour is convex hull, you should fit contour[0] to choose the item
    :return: (x, y) for cv2
    """
    M = cv2.moments(contour)
    head_center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    return head_center


def calc_object_mass_center(roi):
    """

    :param roi:
    :return: (x, y) for cv2
    """
    ys, xs = np.nonzero(roi)

    M00 = np.sum(roi)
    M10 = sum([x * roi[y, x] for x, y in zip(xs, ys)])
    M01 = sum([y * roi[y, x] for x, y in zip(xs, ys)])

    mass_of_center = (int(M10 / M00), int(M01 / M00))
    return mass_of_center


def calc_given_phi_arg(phi, polars):
    return np.argmin([abs(pt[1] - (divmod(phi + math.pi, math.pi * 2)[1] - math.pi)) for pt in polars])


def find_maximum_line_segment(polars):
    """

    :param polars: N x 2, Contour must be entire list, can't be compressed or convex hull. And It's centralized
    :return: the indexes of line_segment
    """

    t1_flag = 0
    t2_flag = calc_given_phi_arg(polars[t1_flag][1] - math.pi, polars)

    # the sum of two point with center, |phi1 - phi2| = pi
    entire_dis = [pt[0] + polars[t2_flag + calc_given_phi_arg(pt[1] - math.pi, polars[t2_flag:])][0] for pt in polars[:t2_flag]]

    # get the pair of maximum distance
    arg_cur_pt = np.argmax(entire_dis)
    arg_ptmate = calc_given_phi_arg(polars[arg_cur_pt][1] - math.pi, polars[t2_flag:]) + t2_flag

    return arg_cur_pt, arg_ptmate


def find_vertical_line_segment(phi, polars):
    """
    Given a pt, find the vertical line segment
    :param phi:
    :param polars:
    :return: the indexes of line_segment
    """

    v1 = calc_given_phi_arg(phi - math.pi / 2, polars)
    v2 = calc_given_phi_arg(phi + math.pi / 2, polars)

    return v1, v2


def calc_corner_pts(center, contour_pts) -> Tuple[CartesianCoordLike, CartesianCoordLike, CartesianCoordLike, CartesianCoordLike]:
    """

    :param center:
    :param contour_pts: N x 1 x 2, Contour must be entire list, can't be compressed or convex hull.
    :return: (x, y), opencv style
    """

    # Set step as 3 for acceleration. N/3 X 1 X 2
    # contour_pts = contour_pts[::3]
    # Centralization
    polars = [calc_cylindrical_rho_phi([pt[0][0] - center[0], pt[0][1] - center[1]]) for pt in contour_pts]

    # farthest_pt_arg = np.argmax([p[0] for p in polars])
    # refer_radian = polars[farthest_pt_arg][1]
    i1, i2 = find_maximum_line_segment(polars)
    y1, y2 = [tuple(contour_pts[i].flatten()) for i in [i1, i2]]
    x1, x2 = [tuple(contour_pts[i].flatten()) for i in find_vertical_line_segment(polars[i1][1], polars)]

    return y1, y2, x1, x2


def solve_linear_function(p1: CartesianCoordLike, p2: CartesianCoordLike) -> Tuple[float, float]:
    """
    to solve k,b for y = k*x + b
    :param p1: CartesianCoordLike, (x, y)
    :param p2: CartesianCoordLike, (x ,y)
    :return: tuple of float, (k, b)
    """

    (x1, y1), (x2, y2) = p1, p2
    k: float = (y1 - y2) / (x1 - x2 + 1e-5)
    b: float = y1 - k * x1

    return k, b


def split_region_by_line(head_region: np.ndarray, k: float, b: float) -> Tuple[List, List]:
    """

    :param head_region: heatmap of head
    :param k: coefficient of first order term
    :param b: constant term
    :return: two parts split by a line, (x, y) style
    """
    ys, xs = np.nonzero(head_region)
    part1, part2 = list(), list()
    for x, y in zip(xs, ys):
        if k * x + b - y > 0:
            part1.append((x, y))
        else:
            part2.append((x, y))
    return part1, part2


def split_binary_parts_by_line_single(pt: Tuple[int, int], line: Tuple[float, float]) -> int:
    """

    :param pt: (x, y) style
    :param line: (k, b) for y line
    :return: the allocating index of part
    """

    x, y = pt
    k, b = line

    return 1 if (y - (k * x + b)) >= 0 else 0


def _split_four_parts_by_xyline(interests: List, pts: Tuple[CartesianCoordLike, CartesianCoordLike, CartesianCoordLike, CartesianCoordLike]):
    """

    :param interests: interest pts : [[x, y], ...]
    :param pts: (y1, y2, x1, x2) to build C.elegans coordinate system
    :param yline: (k, b) for y line
    :param xline: (k, b) for x line
    :return: (y, x) numpy style for 4 parts, and corresponding pts for building
    """

    # initialization
    y1, y2, x1, x2 = pts
    xline = solve_linear_function(x1, x2)
    yline = solve_linear_function(y1, y2)
    parts = [[[], []], [[], []]]

    # split parts
    for x, y in interests:
        y_idx = split_binary_parts_by_line_single((x, y), xline)
        x_idx = split_binary_parts_by_line_single((x, y), yline)
        parts[x_idx][y_idx].append((y, x))

    # pts are consist with parts order
    (anterior_y, posterior_y) = (y1, y2) if split_binary_parts_by_line_single(y1, xline) == 1 else (y2, y1)
    (ventral_x, dorsal_x) = (x1, x2) if split_binary_parts_by_line_single(x1, yline) == 1 else (x2, x1)

    return (anterior_y, posterior_y, ventral_x, dorsal_x), parts


def split_four_parts_by_xyline_4roi(roi: np.ndarray, pts: Tuple[CartesianCoordLike, CartesianCoordLike, CartesianCoordLike, CartesianCoordLike]):
    """

    :param roi: the roi image
    :param pts: (y1, y2, x1, x2) to build C.elegans coordinate system
    :param yline: (k, b) for y line
    :param xline: (k, b) for x line
    :return: (y, x) numpy style for 4 parts, and corresponding pts for building
    """
    (anterior_y, posterior_y, ventral_x, dorsal_x), parts = _split_four_parts_by_xyline(interests = [[x, y] for y, x in zip(*np.nonzero(roi))], pts = pts)
    return (anterior_y, posterior_y, ventral_x, dorsal_x), parts


def split_four_parts_by_xyline_4regions(regions: List, pts: Tuple[CartesianCoordLike, CartesianCoordLike, CartesianCoordLike, CartesianCoordLike]):
    """

    :param regions: [[xmin, ymin, xmax, ymax, z], ...]
    :param pts: (y1, y2, x1, x2) to build C.elegans coordinate system
    :param yline: (k, b) for y line
    :param xline: (k, b) for x line
    :return: (y, x) numpy style for 4 parts, and corresponding pts for building
    """
    (anterior_y, posterior_y, ventral_x, dorsal_x), parts = _split_four_parts_by_xyline(interests = [calc_mean_pt_of_bbox(region)[:2] for region in regions], pts = pts)
    return (anterior_y, posterior_y, ventral_x, dorsal_x), parts


def split_four_parts_by_xyline_4neurons(neurons: dict, pts: Tuple[CartesianCoordLike, CartesianCoordLike, CartesianCoordLike, CartesianCoordLike]):
    """

    :param roi: the roi image
    :param pts: (y1, y2, x1, x2) to build C.elegans coordinate system
    :param yline: (k, b) for y line
    :param xline: (k, b) for x line
    :return: (y, x) numpy style for 4 parts, and corresponding pts for building
    """
    (anterior_y, posterior_y, ventral_x, dorsal_x), parts = _split_four_parts_by_xyline(interests = [calc_mean_pt_of_3d_neuron(neuron)[:2] for neuron in neurons.values()], pts = pts)
    return (anterior_y, posterior_y, ventral_x, dorsal_x), parts


def _calc_direction_xyaxis(pts: Tuple[CartesianCoordLike, CartesianCoordLike, CartesianCoordLike, CartesianCoordLike], parts_sum: List[List[int]]):
    """
    for keep the same positive direction with python,
        we set head direction as the positive y-axis represented by 1 in part list,
               tail direction as the negative y-axis represented by 0 is part list
           and ventral direction as the positive x-axis represented by 1 in part list,
               dorsal direction as the negative x-axis represented by 0 in part list.
    (1, 1): 1 quadrant, (1, 0): 2, (0, 0): 3, (0, 1): 4

    :param pts: [anterior_y, posterior_y, ventral_x, dorsal_x], (x, y) style
    :param parts_sum:
    :return:
    """

    anterior_y, posterior_y, ventral_x, dorsal_x = pts

    # decide positive y-axis direction
    if (parts_sum[1][1] + parts_sum[0][1]) >= (parts_sum[0][0] + parts_sum[1][0]):
        anterior_y, posterior_y = posterior_y, anterior_y
        # parts_sum[0], parts_sum[1] = parts_sum[1][::-1], parts_sum[0][::-1]
    else:
        pass

    # decide positive x-axis direction (ventral direction)
    if (parts_sum[1][1] + parts_sum[1][0]) >= (parts_sum[0][0] + parts_sum[0][1]):
        # parts_sum[0], parts_sum[1] = parts_sum[1], parts_sum[0]
        ventral_x, dorsal_x = dorsal_x, ventral_x
    else:
        pass

    return anterior_y, posterior_y, ventral_x, dorsal_x


def calc_direction_xyaxis_4roi(roi: np.ndarray, pts: Tuple[CartesianCoordLike, CartesianCoordLike, CartesianCoordLike, CartesianCoordLike], parts: List[List]):
    return _calc_direction_xyaxis(pts, [[sum([roi[y, x] for y, x in p]) for p in xpart] for xpart in parts])


def calc_direction_xyaxis_4regions(pts: Tuple[CartesianCoordLike, CartesianCoordLike, CartesianCoordLike, CartesianCoordLike], parts: List[List]):
    return _calc_direction_xyaxis(pts, [[len(p) for p in xpart] for xpart in parts])


def calc_direction_xyaxis_4neurons(pts: Tuple[CartesianCoordLike, CartesianCoordLike, CartesianCoordLike, CartesianCoordLike], parts: List[List]):
    return _calc_direction_xyaxis(pts, [[len(p) for p in xpart] for xpart in parts])


# ----- pre-process --------------------
# ----- General -----
def save_image(path, image, _min = None, _max = None):
    plt.imsave(path, image, cmap = 'inferno', vmin = _min, vmax = _max)


# ----- volume -----
def preprocess_volume(volume, mode, load_path):
    """

    :param volume:
    :param mode: It's a nonnegative number
    :return:
    """

    if mode == 0:
        return volume
    else:
        projection = get_image4processing(volume, is_max = True)
        if mode in (1, 4):
            return [volume,
                    projection]

        elif os.path.isfile(load_path):
            results = extract_preprocessing_json(load_path, projection)
            if mode in (2, 5):
                return [volume,
                        projection,
                        results[0]]
            elif mode in (3, 6):
                return [volume,
                        projection,
                        results[0],
                        results[1]]
        else:
            head_ctr, is_head_region_warning = get_head_contour(projection, is_convex = False)
            convex_head_ctr = [cv2.convexHull(head_ctr)]
            convex_head_binary_region = draw_head_region(projection, convex_head_ctr)
            rect_of_roi = cv2.boundingRect(convex_head_ctr[0])  # (x, y, w, h)
            body_ctrs, noise_ctrs = list(), list()
            # body_ctrs, noise_ctrs = get_tail_contour(projection, convex_head_binary_region, is_convex = False, noise_area = 300.0)
            # body_ctrs = [compress_contour(ctr, 0.05)[0] for ctr in body_ctrs]
            # noise_ctrs = [compress_contour(ctr, 0.01)[0] for ctr in noise_ctrs]
            if mode in (2, 5):
                return [volume,
                        projection,
                        [convex_head_ctr, is_head_region_warning, convex_head_binary_region, rect_of_roi, body_ctrs, noise_ctrs]]

            else:
                roi_in_head = (draw_head_region(projection, [head_ctr]) * projection > (projection.mean() + projection.std())) * projection
                # mass_of_center, anterior_y, posterior_y, ventral_x, dorsal_x = locate_coordinate_pts_by_deeplearning(roi_in_head)
                mass_of_center = calc_object_mass_center(roi_in_head)
                anterior_y, posterior_y, ventral_x, dorsal_x = calc_corner_pts(mass_of_center, head_ctr)
                (anterior_y, posterior_y, ventral_x, dorsal_x), parts = split_four_parts_by_xyline_4roi(roi = roi_in_head, pts = (anterior_y, posterior_y, ventral_x, dorsal_x))
                anterior_y, posterior_y, ventral_x, dorsal_x = calc_direction_xyaxis_4roi(roi = roi_in_head, pts = (anterior_y, posterior_y, ventral_x, dorsal_x), parts = parts)

                if mode in (3, 6):
                    return [volume,
                            projection,
                            [convex_head_ctr, is_head_region_warning, convex_head_binary_region, rect_of_roi, body_ctrs, noise_ctrs],
                            [mass_of_center, anterior_y, posterior_y, ventral_x, dorsal_x]]


def preprocess(params):
    stack_path, mode, z_range, load_path, result_root, name_reg = params
    stack_name = get_stack_name(name_reg, stack_path)
    if not stack_name:
        return

    if mode in (4, 5, 6):
        result_path = os.path.join(result_root, stack_name)
        os.makedirs(result_path, exist_ok = True)

    try:
        vols = load_worm_mat(stack_path, is_normalize = False, z_range = z_range)
    except Exception as e:
        print_warning_message(f"Loading .Mat file problem: {stack_path}  {e}")
        return None

    results = dict()
    for idx, volume in enumerate(vols):
        volume_name = stack_name + "_" + pad_num(idx, 3)
        # main step
        result = preprocess_volume(volume = volume, mode = mode, load_path = os.path.join(load_path, stack_name, f"{pad_num(idx, 3)}.json") if load_path != "" else load_path)
        results[volume_name] = result

        if mode == 4:
            save_image(os.path.join(result_path, f"{pad_num(idx, 3)}.png"), result[1])
        elif mode == 5:
            image_path = f"{pad_num(idx, 3)}.png"
            save_image(os.path.join(result_path, image_path), result[1])
            save_anno_as_json(mode = mode,
                              json_path = os.path.join(result_path, f"{pad_num(idx, 3)}.json"),
                              image_path = image_path,
                              image_height = volume.shape[0],
                              image_width = volume.shape[1],
                              head_ctr = result[2][0],
                              is_head_region_warning = result[2][1],
                              tail_ctrs = result[2][4]
                              )
        elif mode == 6:
            image_path = f"{pad_num(idx, 3)}.png"
            save_image(os.path.join(result_path, image_path), result[1])
            save_anno_as_json(mode = mode,
                              json_path = os.path.join(result_path, f"{pad_num(idx, 3)}.json"),
                              image_path = image_path,
                              image_height = volume.shape[0],
                              image_width = volume.shape[1],
                              head_ctr = result[2][0],
                              is_head_region_warning = result[2][1],
                              tail_ctrs = result[2][4],
                              pts = result[3])

    return stack_name, results


def extract_preprocessing_json(path: str, projection):
    with open(path, 'rb') as file:
        json_file = json.load(file)["shapes"]

    body_ctrs, noise_ctrs = list(), list()
    for item in json_file:
        if item['label'] == 'head':
            convex_head_ctr, is_head_region_warning = [np.array(item["points"], dtype = np.int32)], item['group_id']
            convex_head_binary_region = draw_head_region(projection, convex_head_ctr)
            rect_of_roi = cv2.boundingRect(convex_head_ctr[0])  # (x, y, w, h)
        if item['label'] == 'anterior_y':
            anterior_y = [v for v in item['points'][0]]
        if item['label'] == 'posterior_y':
            posterior_y = [v for v in item['points'][0]]
        if item['label'] == 'ventral_x':
            ventral_x = [v for v in item['points'][0]]
        if item['label'] == 'dorsal_x':
            dorsal_x = [v for v in item['points'][0]]
        if item['label'] == 'mass_of_center':
            mass_of_center = [v for v in item['points'][0]]
        if item['label'] == 'tail':
            body_ctrs.append(np.array(item["points"], dtype = np.int32))
        if item['label'] == 'noise':
            noise_ctrs.append(np.array(item["points"], dtype = np.int32))

    return [[convex_head_ctr, is_head_region_warning, convex_head_binary_region, rect_of_roi, body_ctrs, noise_ctrs], [mass_of_center, anterior_y, posterior_y, ventral_x, dorsal_x]]
