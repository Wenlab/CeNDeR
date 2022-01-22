# -*- coding: utf-8 -*-
#
# @Author   : Yuxiang WU
# @Email    : elephantameler@gmail.com

"""
    Reference: https://pypi.org/project/object-detection-metrics/

"""

import sys
from typing import Dict

from merge.xgraph_alignment import XNeuronAlign


def calc_det_score_wlabel(preds: Dict, labels: Dict, sp: Dict, threshold: float):
    """Calculate recall, precision and F1 score for detection with raw label

    :param preds:
    :param labels:
    :param sp:
    :param threshold:
    :return:
    """

    preds = {name: [r for s in vol_res for r in s] for name, vol_res in preds.items()}
    labels = {name: [[r[1] - sp[name][0], r[2] - sp[name][1], r[3] - sp[name][0], r[4] - sp[name][1], r[0]] for s in vol_res.values() for r in s] for name, vol_res in labels.items()}
    precision, recall, f1_score, results = calc_det_score(preds, labels, threshold)

    return precision, recall, f1_score, results


def calc_det_score(preds: Dict, labels: Dict, threshold: float):
    """Calculate recall, precision and F1 score for detection

    :param preds: {vol_name: [R1, R2, ...]} value is a list of slice regions. R: [xmin, ymin, xmax, ymax, z]
    :param labels: {vol_name: [R1, R2, ... ]}.
    :param threshold: float type
    :return: [precision, recall, f1_score, results]
    """

    preds, labels = preds.copy(), labels.copy()
    results = dict()
    tps, fps, fns = 0, 0, 0
    for vol_name in preds.keys():
        tp = 0
        fp = 0
        pred, label = preds[vol_name], labels[vol_name]
        for r in pred:
            max_iou = sys.float_info.min
            max_idx = -1
            for gt_i, gt in enumerate(label):
                if r[-1] == gt[-1]:
                    iou = XNeuronAlign.calc_iou(r[:4], gt[:4])
                    if iou >= max_iou:
                        max_iou = iou
                        max_idx = gt_i
            if max_iou >= threshold:
                tp += 1
                label.pop(max_idx)
            else:
                fp += 1
        fn = len(label)
        tps, fps, fns = tps + tp, fps + fp, fns + fn
        p, r = tp / (tp + fp), tp / (tp + fn)
        results[vol_name] = [p, r, (2 * p * r) / (p + r + 1e-5)]

    precision = tps / (tps + fps)
    recall = tps / (tps + fns)
    f1_score = (2 * precision * recall) / (precision + recall + 1e-5)

    return precision, recall, f1_score, results
