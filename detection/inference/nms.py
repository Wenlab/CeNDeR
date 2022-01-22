# -*- coding: utf-8 -*-
#
# @Author   : Yuxiang WU
# @Email    : elephantameler@gmail.com

import numpy as np

__all__ = ['non_maximum_suppression_volume_func']


# ----------- NMS  -----------
def non_maximum_suppression_volume_func(poses):
    vol_ctn, bboxes, scores, iou_threshold, number_threshold = poses
    result = _non_maximum_suppression_volume(bboxes, scores, iou_threshold, number_threshold)
    return vol_ctn, result


def _non_maximum_suppression_volume(bboxes, scores, iou_threshold, number_threshold):
    """
    processing candidates per frame
    :param bboxes: list, [xmin, ymin, xmax, ymax, z]
    :param scores: list,
    :return: list, [xmin, ymin, xmax, ymax, z]
    """
    keep, keep_scores = list(), list()
    bboxes, scores = np.array(bboxes, dtype = np.int32), np.array(scores, dtype = np.float32)
    for z in np.unique(bboxes[:, -1]):
        # filter xmax <= xmin or ymax <= ymin situations which are appeared due to lack this constraint in DETECTION result.
        idxes = np.atleast_1d(np.argwhere(np.logical_and(bboxes[:, -1] == z, np.logical_and(bboxes[:, 0] < bboxes[:, 2], bboxes[:, 1] < bboxes[:, 3]))).squeeze())

        if len(idxes) > 0:
            _keep, _score = _non_maximum_suppression_frame(bboxes[idxes], scores[idxes], threshold = iou_threshold)
            keep.extend(_keep)
            keep_scores.extend(_score)

    # thresholding number using descending scores
    keep = np.array([keep[idx] for idx in np.argsort(keep_scores)[-min(number_threshold, len(keep)):][::-1]])
    # transferring type for store
    new_keep = [keep[np.atleast_1d(np.argwhere(keep[:, -1] == z).squeeze())] for z in np.unique(keep[:, -1])]

    return new_keep


def _non_maximum_suppression_frame(bboxes, scores, threshold):
    keep, _score = list(), list()

    x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    while order.size > 0:
        i = order[0]
        keep.append(bboxes[i])
        _score.append(scores[i])
        xx1, yy1 = np.maximum(x1[i], x1[order[1:]]), np.maximum(y1[i], y1[order[1:]])
        xx2, yy2 = np.minimum(x2[i], x2[order[1:]]), np.minimum(y2[i], y2[order[1:]])

        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)  # w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-5)

        order = order[1 + np.atleast_1d(np.argwhere(iou <= threshold).squeeze())]

    return keep, _score
