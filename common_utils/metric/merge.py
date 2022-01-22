# -*- coding: utf-8 -*-
#
# @Author   : Yuxiang WU
# @Email    : elephantameler@gmail.com

"""
    3D semantic segmentation metric (IoU, SEG, MUCov):
    https://www.nature.com/articles/s41540-020-00152-8.pdf
    https://github.com/funalab/QCANet/blob/master/src/tools/evaluation_seg.py
"""

import sys
from typing import Dict, List

from common_utils.prints import print_error_message


def calc_merge_score_wlabel(preds: Dict, labels: Dict, sp: Dict, threshold: float, method: str = "SEG"):
    """Calculate SEG, MUCov, and F1 score like det.

    :param preds:
    :param labels:
    :param sp:
    :param threshold:
    :param method:
    :return:
    """

    labels = {name: {i: [[r[1] - sp[name][0], r[2] - sp[name][1], r[3] - sp[name][0], r[4] - sp[name][1], r[0]] for r in n] for i, n in vol_res.items()} for name, vol_res in labels.items()}
    if method.upper() == "SEG":
        method = calc_SEG_score
    elif method.upper() == "MUCOV":
        method = calc_MUCov_score
    elif method.upper() == "F1SCORE":
        method = calc_3d_neuron_score
    else:
        print_error_message(f" Merge Metric {method} doesn't exist! ")

    score, results = method(preds, labels, threshold)

    return score, results


def calc_intersection_union(a: List, b: List):
    """

    :param a:
    :param b:
    :return:
    """

    inter_area = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union_area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter_area

    return inter_area, union_area


def calc_neuron_iou(a: List, b: List):
    """IoU of two list of regions

    :param a: [R1, R2, ...]. R: [xmin, ymin, xmax, ymax, z]
    :param b: the same as a
    :return:
    """

    a, b = a.copy(), b.copy()
    inter_areas = 0
    union_areas = 0
    for r_a in a:
        flag = False
        for i_b, r_b in enumerate(b):
            if r_a[-1] == r_b[-1]:
                flag = True
                inter_area, union_area = calc_intersection_union(r_a[:4], r_b[:4])
                inter_areas += inter_area
                union_areas += union_area
                b.pop(i_b)
                break
        if not flag:
            union_areas += ((r_a[2] - r_a[0]) * (r_a[3] - r_a[1]))
    union_areas += sum([(r[2] - r[0]) * (r[3] - r[1]) for r in b])
    iou = inter_areas / (union_areas + 1e-5)

    return iou


def calc_3d_neuron_score(preds: Dict, labels: Dict, threshold: float):
    """Merge Metric like IoU.

    :param preds: {vol_name: {id: [R1, ...] ...}}. R: [xmin, ymin, xmax, ymax, z]
    :param labels: the same as results
    :param threshold:
    :return: [precision, recall, f1_score, results]
    """

    preds, labels = preds.copy(), labels.copy()
    results = dict()
    tps, fps, fns = 0, 0, 0
    for vol_name in preds.keys():
        tp, fp = 0, 0
        pred, label = preds[vol_name], labels[vol_name]
        pred_list = sorted(list(pred.values()), key = lambda x: len(x), reverse = True)
        label_list = sorted(list(label.values()), key = lambda x: len(x), reverse = True)

        for n in pred_list:
            max_iou = sys.float_info.min
            max_idx = -1
            for gt_i, gt in enumerate(label_list):
                iou = calc_neuron_iou(n, gt)
                if iou >= max_iou:
                    max_iou = iou
                    max_idx = gt_i
            if max_iou >= threshold:
                tp += 1
                label_list.pop(max_idx)
            else:
                fp += 1
        fn = len(label_list)
        tps, fps, fns = tps + tp, fps + fp, fns + fn
        p, r = tp / (tp + fp), tp / (tp + fn)
        results[vol_name] = [p, r, (2 * p * r) / (p + r + 1e-5)]

    precision = tps / (tps + fps)
    recall = tps / (tps + fns)
    f1_score = (2 * precision * recall) / (precision + recall + 1e-5)

    return [precision, recall, f1_score], results


def calc_SEG_score(preds: Dict, labels: Dict, threshold: float = 0.5):
    """SEG Metric. Like recall metric

    :param preds: {vol_name: {id: [R1, ...] ...}}. R: [xmin, ymin, xmax, ymax, z]
    :param labels: the same as results
    :param threshold:
    :return:
    """

    preds, labels = preds.copy(), labels.copy()
    results = dict()
    ious = 0.0
    num_labels = 0

    for vol_name in preds.keys():
        pred, label = preds[vol_name], labels[vol_name]
        pred_list = sorted(list(pred.values()), key = lambda x: len(x), reverse = True)
        label_list = sorted(list(label.values()), key = lambda x: len(x), reverse = True)
        # recording variable
        vol_ious = 0.0
        num_labels += len(label)

        for n in pred_list:
            max_iou = sys.float_info.min
            max_idx = -1
            for gt_i, gt in enumerate(label_list):
                iou = calc_neuron_iou(n, gt)
                if iou >= max_iou:
                    max_iou = iou
                    max_idx = gt_i
            if max_iou >= threshold:
                vol_ious += max_iou
                label_list.pop(max_idx)
        ious += vol_ious
        results[vol_name] = vol_ious / len(label)
    SEG_score = ious / num_labels

    return SEG_score, results


def calc_MUCov_score(preds: Dict, labels: Dict, threshold: float = 0.5):
    """MUCov Metric. Like precision metric

    :param preds: {vol_name: {id: [R1, ...] ...}}. R: [xmin, ymin, xmax, ymax, z]
    :param labels: the same as results
    :param threshold:
    :return:
    """

    preds, labels = preds.copy(), labels.copy()
    results = dict()
    ious = 0.0
    num_preds = 0

    for vol_name in preds.keys():
        pred, label = preds[vol_name], labels[vol_name]
        pred_list = sorted(list(pred.values()), key = lambda x: len(x), reverse = True)
        label_list = sorted(list(label.values()), key = lambda x: len(x), reverse = True)
        # recording variable
        vol_ious = 0.0
        num_preds += len(pred)

        for gt in label_list:
            max_iou = sys.float_info.min
            max_idx = -1
            for p_i, p in enumerate(pred_list):
                iou = calc_neuron_iou(gt, p)
                if iou >= max_iou:
                    max_iou = iou
                    max_idx = p_i
            if max_iou >= threshold:
                vol_ious += max_iou
                pred_list.pop(max_idx)
        ious += vol_ious
        results[vol_name] = vol_ious / len(pred)
    MUCov_score = ious / num_preds

    return MUCov_score, results
