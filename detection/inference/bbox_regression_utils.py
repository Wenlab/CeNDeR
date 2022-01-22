# -*- coding: utf-8 -*-
#
# @Author   : Yuxiang WU
# @Email    : elephantameler@gmail.com

def calc_anchor_score(anchor: [int, int, int, int, int],
                      gt: [int, int, int, int, int]
                      ) -> float:
    """

    :param anchor: [xmin, ymin, xmax, ymax, z], the first two values represent the upper left pt of the anchor,
                                and the second two represent the width and height of the anchor
    :param gt: values of target are the same as anchor
    :return: score (float type)
    """

    if anchor[-1] != gt[-1]:
        return 0.0

    x1_a, y1_a, x2_a, y2_a = anchor[0], anchor[1], anchor[2], anchor[3]
    x1_gt, y1_gt, x2_gt, y2_gt = gt[0], gt[1], gt[2], gt[3]

    inter_area = max(0, min(x2_a, x2_gt) - max(x1_a, x1_gt)) * max(0, min(y2_a, y2_gt) - max(y1_a, y1_gt))
    union_area = (x2_gt - x1_gt) * (y2_gt - y1_gt) + (x2_a - x1_a) * (y2_a - y1_a) - inter_area

    anchor_score = inter_area / (union_area + 1e-5)

    return anchor_score


def calc_regression_params(peak: [int, int, int],
                           anchor_shifts: [int, int, int, int],
                           gt: [int, int, int, int, int]
                           ) -> [float, float, float, float]:
    """
    Supported that peak pt isn't in the gt.
    Peak pt should be in the gt.

    :param peak: [x, y, z]
    :param anchor_shifts: [upper_h, lower_h, left_w, right_w]
    :param gt: [xmin, ymin, xmax, ymax]
    :return: [upper_delta, lower_delta, left_delta, right_delta]
    """

    x, y, _ = peak
    upper_h, lower_h, left_w, right_w = anchor_shifts
    height, width = (upper_h + lower_h), (left_w + right_w)
    xmin, ymin, xmax, ymax, _ = gt

    upper_delta = (y - ymin) / height
    lower_delta = (ymax - y) / height
    left_delta = (x - xmin) / width
    right_delta = (xmax - x) / width

    return [upper_delta, lower_delta, left_delta, right_delta]


def calc_predicted_bbox(peak: [int, int, int],
                        anchor_shifts: [int, int, int, int],
                        deltas: [float, float, float, float],
                        ) -> [int, int, int, int, int]:
    """
    Supported that peak pt isn't in the gt.

    :param peak: [x, y, z]
    :param anchor_shifts: [upper_h, lower_h, left_w, right_w]
    :param deltas: [upper_delta, lower_delta, left_delta, right_delta]
    :return:
    """
    x, y, z = peak
    upper_h, lower_h, left_w, right_w = anchor_shifts
    height, width = (upper_h + lower_h), (left_w + right_w)
    upper_delta, lower_delta, left_delta, right_delta = deltas

    ymin = round(y - upper_delta * height)
    ymax = round(y + lower_delta * height)
    xmin = round(x - left_delta * width)
    xmax = round(x + right_delta * width)

    return [xmin, ymin, xmax, ymax, z]
